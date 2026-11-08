// HBM-PIM timing model, ported from Samsung's PIMSimulator (DRAMSim2 fork).
//
// PIM kernels ("PIMKernel,..." trace commands) are compiled into a
// pim_program: the per-channel sequence of 32 B DRAM transactions that
// PIMSimulator's host-side PIMKernel issues (park-in, mode changes, CRF
// programming, the compute commands, park-out). Every channel runs the same
// program through an in-order controller inside dram_t (dram_pim.cc) that
// reproduces DRAMSim2's transaction/command queues, barriers and scheduling,
// while a pim_unit tracks the channel's PIM mode and steps the CRF program on
// every column command issued in HAB_PIM mode. No data values are modeled.

#ifndef PIM_H
#define PIM_H

#include <stdint.h>
#include <stdio.h>
#include <memory>
#include <string>
#include <vector>

#include "pim_config.h"

enum class pim_mode { SB = 0, HAB, HAB_PIM };
const char *pim_mode_str(pim_mode m);

// PIM instruction set (PIMSimulator PIMCmd.h)
enum pim_cmd_type {
  PIM_NOP = 0,
  PIM_ADD = 1,
  PIM_MUL = 2,
  PIM_MAC = 3,
  PIM_MAD = 4,
  PIM_MOV = 8,
  PIM_FILL = 9,
  PIM_JUMP = 14,
  PIM_EXIT = 15,
  PIM_NUM_CMD_TYPES = 16
};
const char *pim_cmd_type_str(unsigned type);

enum pim_opd_type {
  PIM_OPD_A_OUT = 0,
  PIM_OPD_M_OUT,
  PIM_OPD_EVEN_BANK,
  PIM_OPD_ODD_BANK,
  PIM_OPD_GRF_A,
  PIM_OPD_GRF_B,
  PIM_OPD_SRF_M,
  PIM_OPD_SRF_A
};

// One 32-bit CRF instruction. decode() only writes the fields that the
// decoded opcode carries, like PIMCmd::fromInt(), so stale fields of a
// reused pim_instr behave exactly as in PIMSimulator.
struct pim_instr {
  unsigned type = PIM_NOP;
  unsigned dst = 0, src0 = 0, src1 = 0, src2 = 0;
  unsigned loop_counter = 0, loop_offset = 0;
  unsigned is_auto = 0, is_relu = 0;
  unsigned dst_idx = 0, src0_idx = 0, src1_idx = 0;

  static pim_instr nop(unsigned repeat);
  static pim_instr jump(int count, unsigned offset);
  static pim_instr exit();
  static pim_instr fill(unsigned dst, unsigned src0, bool relu = false);
  static pim_instr alu(unsigned type, unsigned dst, unsigned src0,
                       unsigned src1, bool is_auto);

  uint32_t encode() const;
  void decode(uint32_t word);
  std::string to_string() const;
};

// what a PIM-session command is for (stats and command trace only)
enum pim_req_kind : uint8_t {
  PIM_REQ_STREAM = 0,  // plain read/write traffic (non-PIM baseline)
  PIM_REQ_PARK,        // park-in / park-out
  PIM_REQ_ABMR,        // SB -> HAB
  PIM_REQ_SBMR,        // HAB -> SB
  PIM_REQ_CRF,         // CRF programming
  PIM_REQ_CTRL,        // control register (HAB <-> HAB_PIM)
  PIM_REQ_GRF,         // host write of GRF_A / GRF_B
  PIM_REQ_COMPUTE,     // bank access that drives the PIM blocks
  PIM_NUM_REQ_KINDS
};
const char *pim_req_kind_str(unsigned kind);

// control register payload carried by PIM_REQ_CTRL writes (the data values
// are not modeled, so the generator passes the bits it would have written)
enum pim_ctrl_bits : uint8_t {
  PIM_CTRL_OP = 1,            // u8[0] bit 0: pim_op (enter HAB_PIM)
  PIM_CTRL_TOGGLE_SHIFT = 1,  // u8[16] bits 0..2: toggle condition
  PIM_CTRL_TOGGLE_MASK = 7,
  PIM_CTRL_GRF_A_ZERO = 1 << 4,  // u8[20]
  PIM_CTRL_GRF_B_ZERO = 1 << 5   // u8[21]
};

// One 32 B DRAM transaction of a PIM session
struct pim_req_t {
  unsigned row;
  unsigned col;
  unsigned bank;
  bool write;
  bool barrier;  // DRAMSim2 "BAR" tag: nothing younger may issue before it
  uint8_t kind;
  uint8_t ctrl;               // control payload of PIM_REQ_CTRL writes
  const uint32_t *crf_words;  // 8 instructions written by a PIM_REQ_CRF write
};

// A run of transactions; programs are kept compact (a GEMV is a few thousand
// segments per channel) and expanded lazily by pim_cursor.
struct pim_segment {
  enum pattern_t : uint8_t {
    LINEAR,      // count commands at (bank, row, col + i), wrapping columns
                 // into the next row like PIMAddrManager::addrGenSafe()
    PARK_SWEEP,  // one command per bank, bank-in-group major (parkIn order)
    STREAM       // sequential Scheme8 traffic: chunk k -> bank k % nbk, ...
  };
  pattern_t pattern;
  bool write;
  bool barrier_after;  // barrier on the last command of the segment
  bool drain_before;   // wait until the channel is idle before issuing
  uint8_t kind;
  uint8_t ctrl;
  int crf_chunk;  // PIM_REQ_CRF: index of the 8-instruction chunk
  unsigned bank, row, col;
  unsigned long long count;
  unsigned long long start;  // STREAM: index of the first chunk
};

class pim_program {
 public:
  std::string name;  // kernel name printed in the stats
  std::vector<pim_segment> segments;
  std::vector<uint32_t> crf;  // CRF image, padded to a multiple of 8
  unsigned nbk = 16;
  unsigned nbkgrp = 4;
  unsigned cols_per_row = 32;
  unsigned trans_queue_size = 64;
  bool is_pim = true;  // false for the plain-traffic stream baselines

  // closed-form expectations checked when the session finishes
  unsigned long long expected_cmds = 0;      // transactions per channel
  unsigned long long expected_pim_cmds = 0;  // CRF-stepping column commands
  unsigned long long expected_pim_on = 0;    // HAB -> HAB_PIM transitions

  void add(pim_segment::pattern_t pattern, bool write, unsigned bank,
           unsigned row, unsigned col, unsigned long long count, uint8_t kind,
           bool barrier_after);
  void add_ctrl(unsigned reg_row, bool pim_op, unsigned toggle_cond,
                bool grf_a_zero, bool grf_b_zero);
  void barrier();  // tag the last transaction added (PIMKernel::addBarrier)
  void drain();    // next transaction waits for an idle channel
  unsigned long long total_cmds() const;

 private:
  bool m_drain_next = false;
  unsigned long long m_since_drain = 0;  // transactions queued since a drain
};

// Walks a pim_program for one channel
class pim_cursor {
 public:
  void reset(const pim_program *prog);
  bool done() const;
  bool at_fence() const;  // the next transaction waits for an idle channel
  void pass_fence();
  bool can_release() const { return !done() && !at_fence(); }
  pim_req_t next();

 private:
  void skip_empty();
  const pim_program *m_prog = nullptr;
  size_t m_seg = 0;
  unsigned long long m_idx = 0;
  unsigned m_row = 0, m_col = 0;  // LINEAR wrap state
  bool m_fence_passed = false;
};

struct pim_stats {
  unsigned long long mode_cycles[3] = {0, 0, 0};  // session DRAM cycles
  unsigned long long n_act = 0, n_pre = 0, n_rd = 0, n_wr = 0;
  unsigned long long n_allbank_act = 0, n_allbank_pre = 0;
  unsigned long long n_allbank_rd = 0, n_allbank_wr = 0;
  unsigned long long n_ref = 0, n_ref_other_rank = 0;
  unsigned long long n_pim_cmds = 0;       // column commands stepping the CRF
  unsigned long long n_reserved_cmds = 0;  // column commands to reserved rows
  unsigned long long crf_visits[PIM_NUM_CMD_TYPES] = {};
  unsigned long long block_ops[PIM_NUM_CMD_TYPES] = {};
  unsigned long long mac_lane_ops = 0;  // FP16 MAC/MAD lane operations
  unsigned long long crf_writes = 0, grf_a_writes = 0, grf_b_writes = 0;
  unsigned long long srf_writes = 0, ctrl_writes = 0;
  unsigned long long sb_to_hab = 0, hab_to_hab_pim = 0, hab_pim_to_hab = 0,
                     hab_to_sb = 0;
  unsigned long long pim_off_before_exit = 0;  // PIM turned off before EXIT
  unsigned long long blocked_cycles = 0;       // work queued but nothing issued

  pim_stats operator-(const pim_stats &o) const;
};

// The PIM side of one DRAM channel: mode machine and CRF control flow
// (PIMSimulator Rank / PIMRank without the data path)
class pim_unit {
 public:
  pim_unit(const pim_config *config, unsigned nbk);

  pim_mode mode() const { return m_mode; }
  bool act_allowed(unsigned bank) const {
    return m_mode == pim_mode::SB || bank < 2;
  }
  bool crf_exited() const { return m_crf_exit; }

  // called when the command is issued on the command bus
  void on_activate(unsigned bank, unsigned row, unsigned col);
  void on_precharge(unsigned bank, unsigned row);
  void on_column(const pim_req_t &req);

  pim_stats &stats() { return m_stats; }
  const pim_stats &stats() const { return m_stats; }

 private:
  bool toggle_cond(const pim_req_t &req) const;
  void control(uint8_t ctrl);
  void step_crf();

  const pim_config *m_config;
  unsigned m_nbk;
  pim_mode m_mode;
  bool m_abmr[4];  // ABMR activates seen on banks 0, 1, 8, 9
  bool m_sbmr[2];  // SBMR precharges seen on banks 0, 1

  std::vector<uint32_t> m_crf;
  int m_pc;
  int m_last_jump_idx, m_jumps_left;
  int m_last_repeat_idx, m_repeats_left;
  bool m_crf_exit;
  bool m_pim_op;
  bool m_toggle_even, m_toggle_odd, m_toggle_ra13h;

  pim_stats m_stats;
};

// Per-channel state of a running PIM kernel (owned by dram_t)
struct pim_session {
  enum phase_t { IDLE = 0, DRAINING, ACTIVE } phase = IDLE;
  std::shared_ptr<const pim_program> prog;
  pim_cursor cursor;
  std::vector<pim_req_t> transq;  // transaction queue (DRAMSim2 order)
  std::vector<pim_req_t> cmdq;    // command queue
  std::vector<unsigned long long> completions;  // min-heap of done cycles
  unsigned long long outstanding = 0;
  bool prefill = false;
  unsigned long long now = 0;  // session DRAM cycle
  unsigned next_bank_pre = 0;
  unsigned long long cmds_issued = 0;  // ACT/PRE/RD/WR/REF, for progress
  unsigned long long transactions = 0;

  unsigned long long start_dram_cycle = 0, end_dram_cycle = 0;
  pim_stats start_stats;   // pim_unit stats when the session started
  pim_stats kernel_stats;  // delta of the last finished session
  bool finished = false;   // a session has finished since the last launch
};

// "PIMKernel,<op>,key=value,..." trace command
struct pim_kernel_desc {
  std::string line;
  std::string op;        // gemv, add, mul, relu, stream
  std::string baseline;  // stream: gemv, add, mul or relu
  unsigned long long out = 0, in = 0, n = 0, batch = 1;
  unsigned long long rd_bytes = 0, wr_bytes = 0;  // custom stream
  unsigned channels = 0;      // channels the data is spread over (0 = all)
  unsigned sim_channels = 0;  // channels simulated (0 = channels)
  bool fix_layout = false;
};

bool parse_pim_kernel_command(const std::string &line, pim_kernel_desc &desc,
                              std::string &error);
std::shared_ptr<pim_program> build_pim_program(const pim_kernel_desc &desc,
                                               const pim_config &config,
                                               unsigned nbk, unsigned nbkgrp,
                                               unsigned n_channels,
                                               std::string &error);

#endif
