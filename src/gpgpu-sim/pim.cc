// HBM-PIM timing model, ported from Samsung's PIMSimulator (DRAMSim2 fork).
// See pim_config.h for an overview.

#include "pim.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <sstream>

#include "../option_parser.h"
#include "dram.h"
#include "gpu-sim.h"

void pim_config::reg_options(OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_pim_enabled", OPT_BOOL, &enabled,
                         "enable the HBM-PIM model in every DRAM channel", "0");
  option_parser_register(opp, "-gpgpu_pim_num_blocks", OPT_UINT32, &num_blocks,
                         "number of PIM blocks per DRAM channel", "8");
  option_parser_register(opp, "-gpgpu_pim_banks_per_block", OPT_UINT32,
                         &banks_per_block,
                         "number of banks sharing one PIM block", "2");
  option_parser_register(opp, "-gpgpu_pim_crf_size", OPT_UINT32, &crf_size,
                         "number of CRF (command register file) entries", "32");
  option_parser_register(opp, "-gpgpu_pim_grf_size", OPT_UINT32, &grf_size,
                         "number of entries in each of GRF_A and GRF_B", "8");
  option_parser_register(opp, "-gpgpu_pim_cols_per_row", OPT_UINT32,
                         &cols_per_row,
                         "number of 32 B columns per DRAM row used by the PIM "
                         "kernel generator",
                         "32");
  option_parser_register(opp, "-gpgpu_pim_reg_row", OPT_UINT32, &reg_row,
                         "PIM register row (control, SRF, CRF, GRF)", "0x3fff");
  option_parser_register(opp, "-gpgpu_pim_abmr_row", OPT_UINT32, &abmr_row,
                         "all-bank mode register row (SB -> HAB)", "0x27ff");
  option_parser_register(opp, "-gpgpu_pim_sbmr_row", OPT_UINT32, &sbmr_row,
                         "single-bank mode register row (HAB -> SB)", "0x2fff");
  option_parser_register(opp, "-gpgpu_pim_park_row", OPT_UINT32, &park_row,
                         "dummy row used by park-in / park-out", "0x2000");
  option_parser_register(
      opp, "-gpgpu_pim_reserved_row_bit", OPT_UINT32, &reserved_row_bit,
      "row address bit that marks a reserved (PIM) row", "13");
  option_parser_register(opp, "-gpgpu_pim_trans_queue_size", OPT_UINT32,
                         &trans_queue_size,
                         "PIM controller transaction queue depth", "64");
  option_parser_register(opp, "-gpgpu_pim_cmd_queue_size", OPT_UINT32,
                         &cmd_queue_size, "PIM controller command queue depth",
                         "64");
  option_parser_register(
      opp, "-gpgpu_pim_cmd_latency", OPT_UINT32, &cmd_latency,
      "command to device latency (tCMD) in DRAM cycles", "1");
  option_parser_register(opp, "-gpgpu_pim_selfcheck", OPT_BOOL, &selfcheck,
                         "check command counts of each PIM kernel against "
                         "their closed-form values",
                         "1");
  option_parser_register(opp, "-gpgpu_pim_align_refresh", OPT_BOOL,
                         &align_refresh,
                         "restart DRAM refresh countdowns when a PIM kernel "
                         "launches",
                         "0");
  option_parser_register(opp, "-gpgpu_pim_cmd_trace", OPT_CSTR, &cmd_trace,
                         "write the DRAM command trace of one PIM channel to "
                         "this file (empty = off)",
                         "");
  option_parser_register(opp, "-gpgpu_pim_cmd_trace_ch", OPT_UINT32,
                         &cmd_trace_ch,
                         "channel traced by -gpgpu_pim_cmd_trace", "0");
}

void pim_config::init(const memory_config *mem_config) {
  if (!enabled) return;

  if (mem_config->simple_dram_model) {
    fprintf(stderr,
            "GPGPU-Sim PIM: -gpgpu_pim_enabled requires the detailed DRAM "
            "model (-gpgpu_simple_dram_model 0)\n");
    exit(1);
  }
  if (mem_config->scheduler_type != DRAM_FRFCFS &&
      mem_config->scheduler_type != DRAM_FIFO) {
    fprintf(stderr, "GPGPU-Sim PIM: unknown DRAM scheduler type\n");
    exit(1);
  }
  if (mem_config->nbk != num_blocks * banks_per_block) {
    fprintf(stderr,
            "GPGPU-Sim PIM: nbk (%u) must equal -gpgpu_pim_num_blocks (%u) x "
            "-gpgpu_pim_banks_per_block (%u)\n",
            mem_config->nbk, num_blocks, banks_per_block);
    exit(1);
  }
  if (banks_per_block != 2) {
    fprintf(stderr,
            "GPGPU-Sim PIM: only 2 banks per PIM block (even/odd bank pair) "
            "are supported\n");
    exit(1);
  }
  if (crf_size == 0 || crf_size % 8 != 0) {
    fprintf(stderr,
            "GPGPU-Sim PIM: -gpgpu_pim_crf_size must be a multiple of 8 (one "
            "32 B column holds 8 instructions)\n");
    exit(1);
  }
  if (!is_reserved_row(reg_row) || !is_reserved_row(abmr_row) ||
      !is_reserved_row(sbmr_row) || !is_reserved_row(park_row)) {
    fprintf(stderr,
            "GPGPU-Sim PIM: the register, ABMR, SBMR and park rows must have "
            "row bit %u set (-gpgpu_pim_reserved_row_bit)\n",
            reserved_row_bit);
    exit(1);
  }

  // Bank group of bank b, as computed by dram_t::get_bankgrp_number()
  auto bank_group = [mem_config](unsigned b) {
    if (mem_config->dram_bnkgrp_indexing_policy == HIGHER_BITS)
      return b >> mem_config->bk_tag_length;
    return b & (mem_config->nbkgrp - 1);
  };
  if (mem_config->nbkgrp > 1 && bank_group(0) != bank_group(1))
    fprintf(stdout,
            "GPGPU-Sim PIM: WARNING banks 0 and 1 are in different bank "
            "groups; HBM-PIM all-bank commands alternate between them, so "
            "use -dram_bnkgrp_indexing_policy 0 to get tCCD_L spacing\n");
  if (mem_config->dram_atom_size != 32)
    fprintf(stdout,
            "GPGPU-Sim PIM: WARNING DRAM atom size is %u B; HBM-PIM commands "
            "move 32 B (busW x BL)\n",
            mem_config->dram_atom_size);
  if (mem_config->dram_bnk_indexing_policy != LINEAR_BK_INDEX)
    fprintf(stdout,
            "GPGPU-Sim PIM: WARNING bank hashing is enabled; it only affects "
            "GPU requests, PIM commands address banks directly\n");
}

////////////////////////////////////////////////////////////////////////////////
// names

const char *pim_mode_str(pim_mode m) {
  switch (m) {
    case pim_mode::SB:
      return "SB";
    case pim_mode::HAB:
      return "HAB";
    case pim_mode::HAB_PIM:
      return "HAB_PIM";
  }
  return "?";
}

const char *pim_cmd_type_str(unsigned type) {
  switch (type) {
    case PIM_NOP:
      return "NOP";
    case PIM_ADD:
      return "ADD";
    case PIM_MUL:
      return "MUL";
    case PIM_MAC:
      return "MAC";
    case PIM_MAD:
      return "MAD";
    case PIM_MOV:
      return "MOV";
    case PIM_FILL:
      return "FILL";
    case PIM_JUMP:
      return "JUMP";
    case PIM_EXIT:
      return "EXIT";
  }
  return "REV";
}

const char *pim_req_kind_str(unsigned kind) {
  static const char *names[PIM_NUM_REQ_KINDS] = {
      "STREAM", "PARK", "SB_TO_HAB", "HAB_TO_SB",
      "CRF",    "CTRL", "GRF",       "COMPUTE"};
  return kind < PIM_NUM_REQ_KINDS ? names[kind] : "?";
}

////////////////////////////////////////////////////////////////////////////////
// PIM ISA (PIMSimulator PIMCmd.cpp)

static uint32_t to_bits(uint32_t val, int len, int pos) {
  return (val & ((1u << len) - 1)) << pos;
}
static uint32_t from_bits(uint32_t val, int len, int pos) {
  return (val >> pos) & ((1u << len) - 1);
}

pim_instr pim_instr::nop(unsigned repeat) {
  pim_instr i;
  i.type = PIM_NOP;
  i.loop_counter = repeat;
  return i;
}

pim_instr pim_instr::jump(int count, unsigned offset) {
  pim_instr i;
  i.type = PIM_JUMP;
  i.loop_counter = (unsigned)count;  // a negative count wraps like PIMCmd
  i.loop_offset = offset;
  return i;
}

pim_instr pim_instr::exit() {
  pim_instr i;
  i.type = PIM_EXIT;
  return i;
}

pim_instr pim_instr::fill(unsigned dst, unsigned src0, bool relu) {
  pim_instr i;
  i.type = PIM_FILL;
  i.dst = dst;
  i.src0 = src0;
  i.is_relu = relu;
  return i;
}

pim_instr pim_instr::alu(unsigned type, unsigned dst, unsigned src0,
                         unsigned src1, bool is_auto) {
  pim_instr i;
  i.type = type;
  i.dst = dst;
  i.src0 = src0;
  i.src1 = src1;
  i.is_auto = is_auto;
  return i;
}

uint32_t pim_instr::encode() const {
  uint32_t val = to_bits(type, 4, 28);
  switch (type) {
    case PIM_EXIT:
      break;
    case PIM_NOP:
      val |= to_bits(loop_counter, 11, 0);
      break;
    case PIM_JUMP:
      val |= to_bits(loop_counter, 17, 11);
      val |= to_bits(loop_offset, 11, 0);
      break;
    case PIM_FILL:
    case PIM_MOV:
      val |= to_bits(dst, 3, 25);
      val |= to_bits(src0, 3, 22);
      val |= to_bits(dst_idx, 4, 8);
      val |= to_bits(src0_idx, 4, 4);
      val |= to_bits(src1_idx, 4, 0);
      val |= to_bits(is_relu, 1, 12);
      break;
    case PIM_MAD:
      val |= to_bits(src2, 3, 16);
      // fall through
    case PIM_ADD:
    case PIM_MUL:
    case PIM_MAC:
      val |= to_bits(dst, 3, 25);
      val |= to_bits(src0, 3, 22);
      val |= to_bits(src1, 3, 19);
      val |= to_bits(is_auto, 1, 15);
      val |= to_bits(dst_idx, 4, 8);
      val |= to_bits(src0_idx, 4, 4);
      val |= to_bits(src1_idx, 4, 0);
      break;
    default:
      break;
  }
  return val;
}

void pim_instr::decode(uint32_t val) {
  type = from_bits(val, 4, 28);
  switch (type) {
    case PIM_EXIT:
      break;
    case PIM_NOP:
      loop_counter = from_bits(val, 11, 0);
      break;
    case PIM_JUMP:
      loop_counter = from_bits(val, 17, 11);
      loop_offset = from_bits(val, 11, 0);
      break;
    case PIM_FILL:
    case PIM_MOV:
      dst = from_bits(val, 3, 25);
      src0 = from_bits(val, 3, 22);
      is_relu = from_bits(val, 1, 12);
      dst_idx = from_bits(val, 4, 8);
      src0_idx = from_bits(val, 4, 4);
      src1_idx = from_bits(val, 4, 0);
      break;
    case PIM_MAD:
      src2 = from_bits(val, 3, 16);
      // fall through
    case PIM_ADD:
    case PIM_MUL:
    case PIM_MAC:
      dst = from_bits(val, 3, 25);
      src0 = from_bits(val, 3, 22);
      src1 = from_bits(val, 3, 19);
      is_auto = from_bits(val, 1, 15);
      dst_idx = from_bits(val, 4, 8);
      src0_idx = from_bits(val, 4, 4);
      src1_idx = from_bits(val, 4, 0);
      break;
    default:
      break;
  }
}

std::string pim_instr::to_string() const {
  static const char *opd[] = {"A_OUT", "M_OUT", "EVEN_BANK", "ODD_BANK",
                              "GRF_A", "GRF_B", "SRF_M",     "SRF_A"};
  std::stringstream ss;
  ss << pim_cmd_type_str(type);
  switch (type) {
    case PIM_NOP:
      ss << " " << loop_counter;
      break;
    case PIM_JUMP:
      ss << " " << loop_counter << " " << loop_offset;
      break;
    case PIM_FILL:
    case PIM_MOV:
      ss << " " << opd[dst & 7] << " " << opd[src0 & 7]
         << (is_relu ? " relu" : "");
      break;
    case PIM_ADD:
    case PIM_MUL:
    case PIM_MAC:
    case PIM_MAD:
      ss << " " << opd[dst & 7] << " " << opd[src0 & 7] << " " << opd[src1 & 7]
         << (is_auto ? " auto" : "");
      break;
    default:
      break;
  }
  return ss.str();
}

////////////////////////////////////////////////////////////////////////////////
// programs

void pim_program::add(pim_segment::pattern_t pattern, bool write, unsigned bank,
                      unsigned row, unsigned col, unsigned long long count,
                      uint8_t kind, bool barrier_after) {
  if (count == 0) return;
  pim_segment s;
  s.pattern = pattern;
  s.write = write;
  s.barrier_after = false;
  s.drain_before = m_drain_next;
  s.kind = kind;
  s.ctrl = 0;
  s.crf_chunk = -1;
  s.bank = bank;
  s.row = row;
  s.col = col;
  s.count = count;
  s.start = 0;
  segments.push_back(s);
  if (m_drain_next) m_since_drain = 0;
  m_drain_next = false;
  m_since_drain += count;
  if (barrier_after) barrier();
}

void pim_program::add_ctrl(unsigned reg_row, bool pim_op, unsigned toggle_cond,
                           bool grf_a_zero, bool grf_b_zero) {
  // PIMKernel::changePIMMode(HAB <-> HAB_PIM): one write of the control
  // register (column 0 of the PIM register row) through bank 0, then a
  // barrier
  add(pim_segment::LINEAR, true, 0, reg_row, 0, 1, PIM_REQ_CTRL, false);
  pim_segment &s = segments.back();
  s.ctrl = (pim_op ? PIM_CTRL_OP : 0) |
           ((toggle_cond & PIM_CTRL_TOGGLE_MASK) << PIM_CTRL_TOGGLE_SHIFT) |
           (grf_a_zero ? PIM_CTRL_GRF_A_ZERO : 0) |
           (grf_b_zero ? PIM_CTRL_GRF_B_ZERO : 0);
  barrier();
}

void pim_program::barrier() {
  // DRAMSim2 MemorySystem::addBarrier() tags the newest queued transaction,
  // except when the controller queue is exactly full and nothing is pending
  // yet, in which case the barrier is silently dropped
  if (segments.empty() || m_since_drain == trans_queue_size) return;
  segments.back().barrier_after = true;
}

void pim_program::drain() { m_drain_next = true; }

unsigned long long pim_program::total_cmds() const {
  unsigned long long n = 0;
  for (const pim_segment &s : segments) n += s.count;
  return n;
}

void pim_cursor::reset(const pim_program *prog) {
  m_prog = prog;
  m_seg = 0;
  m_idx = 0;
  m_fence_passed = false;
  skip_empty();
}

void pim_cursor::skip_empty() {
  while (m_prog && m_seg < m_prog->segments.size() &&
         m_prog->segments[m_seg].count == 0)
    m_seg++;
}

bool pim_cursor::done() const {
  return !m_prog || m_seg >= m_prog->segments.size();
}

bool pim_cursor::at_fence() const {
  return !done() && m_idx == 0 && m_prog->segments[m_seg].drain_before &&
         !m_fence_passed;
}

void pim_cursor::pass_fence() { m_fence_passed = true; }

pim_req_t pim_cursor::next() {
  assert(can_release());
  const pim_segment &s = m_prog->segments[m_seg];
  pim_req_t r;
  r.write = s.write;
  r.barrier = false;
  r.kind = s.kind;
  r.ctrl = s.ctrl;
  r.crf_words = s.crf_chunk >= 0 ? &m_prog->crf[8 * s.crf_chunk] : nullptr;
  switch (s.pattern) {
    case pim_segment::LINEAR:
      // PIMAddrManager::addrGenSafe(): columns past the end of the row
      // continue in the next row
      if (m_idx == 0) {
        m_row = s.row;
        m_col = s.col;
      }
      while (m_col >= m_prog->cols_per_row) {
        m_row++;
        m_col -= m_prog->cols_per_row;
      }
      r.bank = s.bank;
      r.row = m_row;
      r.col = m_col;
      m_col++;
      break;
    case pim_segment::PARK_SWEEP: {
      // PIMKernel::parkIn(): bank-in-group outer loop, bank group inner loop
      unsigned banks_per_group = m_prog->nbk / m_prog->nbkgrp;
      unsigned ba = m_idx / m_prog->nbkgrp;
      unsigned bg = m_idx % m_prog->nbkgrp;
      r.bank = ba + bg * banks_per_group;
      r.row = s.row;
      r.col = s.col;
      break;
    }
    case pim_segment::STREAM: {
      // consecutive 32 B chunks of one channel under Scheme8 addressing
      unsigned long long k = s.start + m_idx;
      r.bank = k % m_prog->nbk;
      r.col = (k / m_prog->nbk) % m_prog->cols_per_row;
      r.row = k / (m_prog->nbk * m_prog->cols_per_row);
      break;
    }
  }
  m_idx++;
  if (m_idx == s.count) {
    r.barrier = s.barrier_after;
    m_seg++;
    m_idx = 0;
    m_fence_passed = false;
    skip_empty();
  }
  return r;
}

////////////////////////////////////////////////////////////////////////////////
// stats

pim_stats pim_stats::operator-(const pim_stats &o) const {
  pim_stats d = *this;
  for (int i = 0; i < 3; i++) d.mode_cycles[i] -= o.mode_cycles[i];
  d.n_act -= o.n_act;
  d.n_pre -= o.n_pre;
  d.n_rd -= o.n_rd;
  d.n_wr -= o.n_wr;
  d.n_allbank_act -= o.n_allbank_act;
  d.n_allbank_pre -= o.n_allbank_pre;
  d.n_allbank_rd -= o.n_allbank_rd;
  d.n_allbank_wr -= o.n_allbank_wr;
  d.n_ref -= o.n_ref;
  d.n_ref_other_rank -= o.n_ref_other_rank;
  d.n_pim_cmds -= o.n_pim_cmds;
  d.n_reserved_cmds -= o.n_reserved_cmds;
  for (int i = 0; i < PIM_NUM_CMD_TYPES; i++) {
    d.crf_visits[i] -= o.crf_visits[i];
    d.block_ops[i] -= o.block_ops[i];
  }
  d.mac_lane_ops -= o.mac_lane_ops;
  d.crf_writes -= o.crf_writes;
  d.grf_a_writes -= o.grf_a_writes;
  d.grf_b_writes -= o.grf_b_writes;
  d.srf_writes -= o.srf_writes;
  d.ctrl_writes -= o.ctrl_writes;
  d.sb_to_hab -= o.sb_to_hab;
  d.hab_to_hab_pim -= o.hab_to_hab_pim;
  d.hab_pim_to_hab -= o.hab_pim_to_hab;
  d.hab_to_sb -= o.hab_to_sb;
  d.pim_off_before_exit -= o.pim_off_before_exit;
  d.blocked_cycles -= o.blocked_cycles;
  return d;
}

////////////////////////////////////////////////////////////////////////////////
// PIM unit: mode machine (PIMSimulator Rank) and CRF control flow (PIMRank)

pim_unit::pim_unit(const pim_config *config, unsigned nbk)
    : m_config(config),
      m_nbk(nbk),
      m_mode(pim_mode::SB),
      m_crf(config->crf_size, 0),
      m_pc(0),
      m_last_jump_idx(-1),
      m_jumps_left(-1),
      m_last_repeat_idx(-1),
      m_repeats_left(-1),
      m_crf_exit(false),
      m_pim_op(false),
      m_toggle_even(true),
      m_toggle_odd(true),
      m_toggle_ra13h(false) {
  for (bool &b : m_abmr) b = false;
  for (bool &b : m_sbmr) b = false;
}

void pim_unit::on_activate(unsigned bank, unsigned row, unsigned col) {
  // SB -> HAB once the ABMR row has been activated (column 0x1f) in banks
  // 0, 1, 8 and 9 (only 0 and 1 with two banks)
  if (m_mode != pim_mode::SB || row != m_config->abmr_row || col != 0x1f)
    return;
  if (bank == 0) m_abmr[0] = true;
  if (bank == 1) m_abmr[1] = true;
  if (bank == 8) m_abmr[2] = true;
  if (bank == 9) m_abmr[3] = true;
  bool done =
      m_abmr[0] && m_abmr[1] && (m_nbk <= 2 || (m_abmr[2] && m_abmr[3]));
  if (done) {
    for (bool &b : m_abmr) b = false;
    m_mode = pim_mode::HAB;
    m_stats.sb_to_hab++;
  }
}

void pim_unit::on_precharge(unsigned bank, unsigned row) {
  // HAB -> SB once the SBMR row has been precharged in banks 0 and 1
  if (m_mode != pim_mode::HAB || row != m_config->sbmr_row) return;
  if (bank == 0) m_sbmr[0] = true;
  if (bank == 1) m_sbmr[1] = true;
  if (m_sbmr[0] && m_sbmr[1]) {
    m_sbmr[0] = m_sbmr[1] = false;
    m_mode = pim_mode::SB;
    m_stats.hab_to_sb++;
  }
}

bool pim_unit::toggle_cond(const pim_req_t &req) const {
  if (!m_pim_op || m_crf_exit) return false;
  if (!m_toggle_ra13h && m_config->is_reserved_row(req.row)) return false;
  return (m_toggle_even && (req.bank & 1) == 0) ||
         (m_toggle_odd && (req.bank & 1) == 1);
}

void pim_unit::control(uint8_t ctrl) {
  m_pim_op = ctrl & PIM_CTRL_OP;
  unsigned toggle = (ctrl >> PIM_CTRL_TOGGLE_SHIFT) & PIM_CTRL_TOGGLE_MASK;
  m_toggle_even = !(toggle & 1);
  m_toggle_odd = !(toggle & 2);
  m_toggle_ra13h = toggle & 4;
  if (m_pim_op) {
    if (m_mode != pim_mode::HAB_PIM) m_stats.hab_to_hab_pim++;
    m_mode = pim_mode::HAB_PIM;
    m_pc = 0;
    m_last_jump_idx = m_jumps_left = m_last_repeat_idx = m_repeats_left = -1;
    m_crf_exit = false;
  } else {
    if (m_mode == pim_mode::HAB_PIM) {
      m_stats.hab_pim_to_hab++;
      if (!m_crf_exit) m_stats.pim_off_before_exit++;
    }
    m_mode = pim_mode::HAB;
  }
}

void pim_unit::on_column(const pim_req_t &req) {
  if (m_config->is_reserved_row(req.row)) m_stats.n_reserved_cmds++;
  if (m_mode == pim_mode::SB) return;

  if (m_mode == pim_mode::HAB_PIM && toggle_cond(req)) {
    m_stats.n_pim_cmds++;
    step_crf();
    return;
  }

  // HAB (or a non-triggering access in HAB_PIM): host <-> PIM registers
  if (req.write && req.row == m_config->reg_row) {
    if (req.col == 0x00) {
      m_stats.ctrl_writes++;
      control(req.ctrl);
    }
    if (req.col >= 0x08 && req.col <= 0x0f) {
      m_stats.grf_a_writes++;
    } else if (req.col >= 0x18 && req.col <= 0x1f) {
      m_stats.grf_b_writes++;
    } else if (req.col >= 0x04 && req.col <= 0x07) {
      m_stats.crf_writes++;
      unsigned base = (req.col - 0x04) * 8;
      if (req.crf_words && base + 8 <= m_crf.size())
        std::copy(req.crf_words, req.crf_words + 8, m_crf.begin() + base);
    } else if (req.col == 0x01) {
      m_stats.srf_writes++;
    }
  }
}

// PIMRank::doPIM() without the data path: a JUMP costs no command, FILL and
// auto-indexed instructions repeat for 8 commands, NOP k for k+1 commands
void pim_unit::step_crf() {
  const int crf_size = m_crf.size();
  pim_instr cmd;  // reused across the loop like PIMRank's cCmd
  do {
    if (m_pc < 0 || m_pc >= crf_size) {
      m_crf_exit = true;  // ran off the CRF
      break;
    }
    cmd.decode(m_crf[m_pc]);
    m_stats.crf_visits[cmd.type]++;
    if (cmd.type == PIM_EXIT) {
      m_crf_exit = true;
      break;
    } else if (cmd.type == PIM_JUMP) {
      if (m_last_jump_idx != m_pc && (int)cmd.loop_counter > 0) {
        m_last_jump_idx = m_pc;
        m_jumps_left = cmd.loop_counter;
      }
      if (m_jumps_left > 0) {
        m_pc -= cmd.loop_offset;
        m_jumps_left--;
      }
    } else {
      int repeat = -1;
      if (cmd.type == PIM_FILL || cmd.is_auto)
        repeat = 8 - 1;
      else if (cmd.type == PIM_NOP)
        repeat = cmd.loop_counter;
      if (repeat >= 0) {
        if (m_last_repeat_idx != m_pc) {
          m_last_repeat_idx = m_pc;
          m_repeats_left = repeat;
        }
        if (m_repeats_left > 0) {
          m_pc -= 1;
          m_repeats_left--;
        } else {
          m_last_repeat_idx = -1;
        }
      }
      m_stats.block_ops[cmd.type] += m_config->num_blocks;
      if (cmd.type == PIM_MAC || cmd.type == PIM_MAD)
        m_stats.mac_lane_ops += m_config->num_blocks * 16;  // FP16 lanes
    }
    m_pc++;
    if (m_pc >= 0 && m_pc < crf_size) {
      pim_instr next;
      next.decode(m_crf[m_pc]);
      if (next.type == PIM_EXIT) m_crf_exit = true;
    }
  } while (cmd.type == PIM_JUMP);
}

////////////////////////////////////////////////////////////////////////////////
// "PIMKernel,<op>,key=value,..." commands

static bool parse_u64(const std::string &s, unsigned long long &v) {
  if (s.empty()) return false;
  char *end = nullptr;
  v = strtoull(s.c_str(), &end, 0);
  return end && *end == '\0';
}

bool parse_pim_kernel_command(const std::string &line, pim_kernel_desc &desc,
                              std::string &error) {
  desc = pim_kernel_desc();
  desc.line = line;
  std::vector<std::string> tok;
  std::stringstream ss(line);
  std::string t;
  while (std::getline(ss, t, ',')) {
    t.erase(0, t.find_first_not_of(" \t\r"));
    t.erase(t.find_last_not_of(" \t\r") + 1);
    if (!t.empty()) tok.push_back(t);
  }
  if (tok.size() < 2 || tok[0] != "PIMKernel") {
    error = "expected PIMKernel,<op>,key=value,...";
    return false;
  }
  desc.op = tok[1];
  for (size_t i = 2; i < tok.size(); i++) {
    size_t eq = tok[i].find('=');
    if (eq == std::string::npos) {
      error = "argument '" + tok[i] + "' is not key=value";
      return false;
    }
    std::string key = tok[i].substr(0, eq), val = tok[i].substr(eq + 1);
    unsigned long long v = 0;
    if (key == "baseline") {
      desc.baseline = val;
      continue;
    }
    if (!parse_u64(val, v)) {
      error = "bad value for '" + key + "'";
      return false;
    }
    if (key == "out")
      desc.out = v;
    else if (key == "in")
      desc.in = v;
    else if (key == "n")
      desc.n = v;
    else if (key == "batch")
      desc.batch = v;
    else if (key == "rd_bytes")
      desc.rd_bytes = v;
    else if (key == "wr_bytes")
      desc.wr_bytes = v;
    else if (key == "channels")
      desc.channels = v;
    else if (key == "sim_channels")
      desc.sim_channels = v;
    else if (key == "fix_layout")
      desc.fix_layout = v;
    else {
      error = "unknown argument '" + key + "'";
      return false;
    }
  }
  return true;
}

namespace {

// Host-side PIM kernel generator, ported from PIMSimulator's
// tests/PIMKernel.cpp and tests/PIMCmdGen.h (per channel, one PIM rank)
class pim_program_builder {
 public:
  pim_program_builder(pim_program &prog, const pim_config &config)
      : p(prog), cfg(config) {}

  void park() {
    p.add(pim_segment::PARK_SWEEP, false, 0, cfg.park_row, 0, p.nbk,
          PIM_REQ_PARK, false);
  }
  void park_in() {
    p.barrier();
    park();
    p.barrier();
  }
  void park_out() {
    park();
    p.barrier();
  }
  void sb_to_hab() {
    p.add(pim_segment::LINEAR, true, 0, cfg.abmr_row, 0x1f, 1, PIM_REQ_ABMR,
          false);
    p.add(pim_segment::LINEAR, true, 1, cfg.abmr_row, 0x1f, 1, PIM_REQ_ABMR,
          false);
    if (p.nbk >= 2) {
      p.add(pim_segment::LINEAR, true, 8, cfg.abmr_row, 0x1f, 1, PIM_REQ_ABMR,
            false);
      p.add(pim_segment::LINEAR, true, 9, cfg.abmr_row, 0x1f, 1, PIM_REQ_ABMR,
            false);
    }
    p.barrier();
  }
  void hab_to_sb() {
    p.add(pim_segment::LINEAR, true, 0, cfg.sbmr_row, 0x1f, 1, PIM_REQ_SBMR,
          false);
    p.add(pim_segment::LINEAR, true, 1, cfg.sbmr_row, 0x1f, 1, PIM_REQ_SBMR,
          false);
    p.barrier();
  }
  bool program_crf(const std::vector<pim_instr> &cmds, std::string &error) {
    if (cmds.size() > cfg.crf_size) {
      error = "CRF program does not fit in -gpgpu_pim_crf_size";
      return false;
    }
    unsigned chunks = (cmds.size() + 7) / 8;
    p.crf.assign(chunks * 8, pim_instr::nop(0).encode());
    for (size_t i = 0; i < cmds.size(); i++) p.crf[i] = cmds[i].encode();
    for (unsigned c = 0; c < chunks; c++) {
      p.add(pim_segment::LINEAR, true, 1, cfg.reg_row, 0x4 + c, 1, PIM_REQ_CRF,
            false);
      p.segments.back().crf_chunk = c;
    }
    p.barrier();
    return true;
  }
  void cmds(bool write, unsigned bank, unsigned row, unsigned col,
            unsigned count, uint8_t kind) {
    p.add(pim_segment::LINEAR, write, bank, row, col, count, kind, true);
  }

 private:
  pim_program &p;
  const pim_config &cfg;
};

unsigned long long ceil_div(unsigned long long a, unsigned long long b) {
  return (a + b - 1) / b;
}

}  // namespace

std::shared_ptr<pim_program> build_pim_program(const pim_kernel_desc &d,
                                               const pim_config &cfg,
                                               unsigned nbk, unsigned nbkgrp,
                                               unsigned n_channels,
                                               std::string &error) {
  auto prog = std::make_shared<pim_program>();
  pim_program &p = *prog;
  p.nbk = nbk;
  p.nbkgrp = nbkgrp;
  p.cols_per_row = cfg.cols_per_row;
  p.trans_queue_size = cfg.trans_queue_size;
  pim_program_builder b(p, cfg);

  const unsigned nch = d.channels ? d.channels : n_channels;
  const unsigned grf = cfg.grf_size;  // GRF_A = GRF_B = 8 entries
  const unsigned bytes_per_elem = 2;  // FP16
  const unsigned elems_per_burst = 16;
  std::stringstream name;

  if (d.op == "gemv") {
    if (!d.out || !d.in || !d.batch) {
      error = "gemv needs out=, in= (and batch=)";
      return nullptr;
    }
    // PIMKernel::executeGemv() (no adder tree)
    const unsigned long long in_bursts = ceil_div(d.in, elems_per_burst);
    const unsigned long long total_blocks = cfg.num_blocks * nch;
    const int num_out_tiles = ceil(((double)d.out / total_blocks) / grf);
    const int num_in_tiles = ceil((double)in_bursts / grf);
    const int num_batch = d.batch;
    const int jump_even = grf * ceil((double)num_in_tiles / 2) - 1;
    const int jump_odd = grf * (num_in_tiles / 2) - 1;
    if (num_in_tiles == 1)
      fprintf(stdout,
              "GPGPU-Sim PIM: WARNING gemv with a single input tile encodes "
              "an odd-bank JUMP count of -1 (as PIMSimulator does); the CRF "
              "program will not reach EXIT\n");

    std::vector<pim_instr> crf = {
        pim_instr::alu(PIM_MAC, PIM_OPD_GRF_B, PIM_OPD_GRF_A, PIM_OPD_EVEN_BANK,
                       true),
        pim_instr::jump(jump_even, 2),
        pim_instr::alu(PIM_MAC, PIM_OPD_GRF_B, PIM_OPD_GRF_A, PIM_OPD_ODD_BANK,
                       true),
        pim_instr::jump(jump_odd, 2),
        pim_instr::nop(7),
        pim_instr::exit()};

    b.park_in();
    b.sb_to_hab();
    if (!b.program_crf(crf, error)) return nullptr;
    for (int j = 0; j < num_out_tiles; j++) {
      for (int bt = 0; bt < num_batch; bt++) {
        // HAB -> HAB_PIM resets the PC and zeroes GRF_B
        p.add_ctrl(cfg.reg_row, true, 0, false, true);
        const unsigned result_col =
            num_out_tiles * num_in_tiles / 2 * grf * grf +
            (d.fix_layout ? j * num_batch + bt : j + bt) * grf;
        auto compute = [&](int i, unsigned bank) {
          // input tile to GRF_A, then 8 x 8 MACs against the weight rows
          b.cmds(true, 1, cfg.reg_row, 0x8, grf, PIM_REQ_GRF);
          unsigned col = (grf * grf) * (i / 2 + j * num_in_tiles / 2);
          for (unsigned c = 0; c < grf * grf; c += grf)
            b.cmds(false, bank, 0, col + c, grf, PIM_REQ_COMPUTE);
        };
        for (int i = 0; i < num_in_tiles; i += 2) compute(i, 0);
        for (int i = 1; i < num_in_tiles; i += 2) compute(i, 1);
        b.cmds(true, 1, 0, result_col, grf, PIM_REQ_COMPUTE);  // GRF_B->bank
        p.add_ctrl(cfg.reg_row, false, 0, false, false);
      }
    }
    b.hab_to_sb();
    b.park_out();

    p.expected_cmds = 39 + (unsigned long long)num_out_tiles * num_batch *
                               (10 + 72ull * num_in_tiles);
    p.expected_pim_cmds = (unsigned long long)num_out_tiles * num_batch *
                          (64ull * num_in_tiles + 8);
    p.expected_pim_on = (unsigned long long)num_out_tiles * num_batch;
    name << "PIMKernel_gemv_" << d.out << "x" << d.in << "_b" << d.batch;
  } else if (d.op == "add" || d.op == "mul" || d.op == "relu") {
    if (!d.n) {
      error = d.op + " needs n=";
      return nullptr;
    }
    // PIMKernel::executeEltwise()
    const unsigned long long dim = d.batch * ceil_div(d.n, elems_per_burst);
    const unsigned long long num_tile = dim / (nbk * nch * grf);
    if (num_tile == 0) {
      error =
          "n is too small for one tile (16 elements x banks x channels x "
          "8)";
      return nullptr;
    }
    const bool relu = d.op == "relu";
    const unsigned alu = d.op == "add" ? PIM_ADD : PIM_MUL;
    std::vector<pim_instr> crf;
    if (relu) {
      crf = {pim_instr::fill(PIM_OPD_GRF_A, PIM_OPD_EVEN_BANK, true),
             pim_instr::nop(7),
             pim_instr::fill(PIM_OPD_GRF_B, PIM_OPD_ODD_BANK, true),
             pim_instr::nop(7)};
    } else {
      crf = {pim_instr::fill(PIM_OPD_GRF_A, PIM_OPD_EVEN_BANK),
             pim_instr::alu(alu, PIM_OPD_GRF_A, PIM_OPD_GRF_A,
                            PIM_OPD_EVEN_BANK, true),
             pim_instr::nop(7),
             pim_instr::fill(PIM_OPD_GRF_B, PIM_OPD_ODD_BANK),
             pim_instr::alu(alu, PIM_OPD_GRF_B, PIM_OPD_GRF_B, PIM_OPD_ODD_BANK,
                            true),
             pim_instr::nop(7)};
    }
    if (num_tile - 1 != 0)
      crf.push_back(pim_instr::jump(num_tile - 1, crf.size() + 1));
    crf.push_back(pim_instr::exit());

    const unsigned in0_row = 0, in1_row = 128, result_row = 256;
    b.park_in();
    b.sb_to_hab();
    if (!b.program_crf(crf, error)) return nullptr;
    p.add_ctrl(cfg.reg_row, true, 0, false, false);
    for (unsigned long long i = 0; i < num_tile; i++) {
      unsigned c = grf * i;
      if (relu) {
        b.cmds(false, 0, in0_row, c, grf, PIM_REQ_COMPUTE);
        b.cmds(true, 0, result_row, c, grf, PIM_REQ_COMPUTE);
        b.cmds(false, 1, in0_row, c, grf, PIM_REQ_COMPUTE);
        b.cmds(true, 1, result_row, c, grf, PIM_REQ_COMPUTE);
      } else {
        for (unsigned bank = 0; bank < 2; bank++) {
          b.cmds(false, bank, in0_row, c, grf, PIM_REQ_COMPUTE);
          b.cmds(false, bank, in1_row, c, grf, PIM_REQ_COMPUTE);
          b.cmds(true, bank, result_row, c, grf, PIM_REQ_COMPUTE);
        }
      }
    }
    p.add_ctrl(cfg.reg_row, false, 0, false, false);
    b.hab_to_sb();
    b.park_out();

    const unsigned per_tile = relu ? 32 : 48;
    p.expected_cmds = 41 + per_tile * num_tile;
    p.expected_pim_cmds = per_tile * num_tile;
    p.expected_pim_on = 1;
    name << "PIMKernel_" << d.op << "_" << d.n;
  } else if (d.op == "stream") {
    // plain read-then-write traffic, the non-PIM baseline of PIMBench
    // (PIMBenchTestCase::genMemTraffic): sequential 32 B transactions
    // interleaved over the channels, reads drained before the writes
    unsigned long long rd = d.rd_bytes, wr = d.wr_bytes;
    if (d.baseline == "gemv") {
      rd = (d.out * d.in + d.in * d.batch) * bytes_per_elem;
      wr = d.out * d.batch * bytes_per_elem;
    } else if (d.baseline == "add" || d.baseline == "mul") {
      rd = 2 * d.n * d.batch * bytes_per_elem;
      wr = d.n * d.batch * bytes_per_elem;
    } else if (d.baseline == "relu") {
      rd = d.n * d.batch * bytes_per_elem;
      wr = d.n * d.batch * bytes_per_elem;
    } else if (!d.baseline.empty()) {
      error = "unknown stream baseline '" + d.baseline + "'";
      return nullptr;
    }
    if (!rd && !wr) {
      error = "stream needs baseline= or rd_bytes=/wr_bytes=";
      return nullptr;
    }
    // chunks of channel 0 within a range of global 32 B chunks
    const unsigned long long rd_chunks = ceil_div(rd, 32);
    const unsigned long long wr_chunks = ceil_div(wr, 32);
    auto first = [nch](unsigned long long k) { return ceil_div(k, nch); };
    auto stream = [&p](bool write, unsigned long long start,
                       unsigned long long count) {
      if (!count) return;
      p.add(pim_segment::STREAM, write, 0, 0, 0, count, PIM_REQ_STREAM, false);
      p.segments.back().start = start;
    };
    stream(false, first(0), first(rd_chunks) - first(0));
    p.drain();
    stream(true, first(rd_chunks),
           first(rd_chunks + wr_chunks) - first(rd_chunks));
    p.is_pim = false;
    p.expected_cmds = p.total_cmds();
    name << "PIMKernel_stream";
    if (!d.baseline.empty()) {
      name << "_" << d.baseline;
      if (d.baseline == "gemv")
        name << "_" << d.out << "x" << d.in;
      else
        name << "_" << d.n;
    } else {
      name << "_rd" << rd << "_wr" << wr;
    }
  } else {
    error = "unknown PIM kernel '" + d.op + "' (gemv, add, mul, relu, stream)";
    return nullptr;
  }

  p.name = name.str();
  if (p.is_pim && p.total_cmds() != p.expected_cmds) {
    error =
        "internal error: generated command count does not match the "
        "closed form";
    return nullptr;
  }
  return prog;
}
