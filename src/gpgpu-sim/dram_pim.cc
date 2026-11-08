// HBM-PIM: in-order controller that runs a PIM program on one DRAM channel.
//
// This is a port of the controller in Samsung's PIMSimulator (DRAMSim2's
// MemorySystem / MemoryController / CommandQueue with PIM barriers), so that
// PIM kernel cycle counts match PIMSimulator's PIMBench. Bank timing reuses
// dram_t's countdown counters: a counter set to d when a command issues at
// cycle t allows the dependent command from cycle t + d, which is exactly
// DRAMSim2's "next* = max(next*, now + d)" rule. Mode-dependent behavior
// (all-bank commands, CRF stepping) lives in pim_unit (pim.cc).

#include <assert.h>
#include <algorithm>
#include <functional>

#include "dram.h"
#include "dram_sched.h"
#include "gpu-sim.h"
#include "pim.h"

void dram_t::pim_launch(std::shared_ptr<const pim_program> prog) {
  assert(m_pim_sess && m_pim_sess->phase == pim_session::IDLE);
  m_pim_sess->prog = prog;
  m_pim_sess->phase = pim_session::DRAINING;
  m_pim_sess->finished = false;
}

bool dram_t::pim_busy() const {
  return m_pim_sess && m_pim_sess->phase != pim_session::IDLE;
}

unsigned long long dram_t::pim_progress() const {
  return m_pim_sess ? m_pim_sess->cmds_issued : 0;
}

// no GPU request left anywhere inside dram_t
bool dram_t::pim_gpu_drained() const {
  if (m_config->scheduler_type == DRAM_FRFCFS &&
      (m_frfcfs_scheduler->num_pending() ||
       m_frfcfs_scheduler->num_write_pending()))
    return false;
  if (!mrqq->empty() || m_rwq_requests) return false;
  for (unsigned b = 0; b < m_config->nbk; b++)
    if (bk[b]->mrq) return false;
  return true;
}

void dram_t::pim_start_session() {
  pim_session &s = *m_pim_sess;
  s.phase = pim_session::ACTIVE;
  s.cursor.reset(s.prog.get());
  s.transq.clear();
  s.cmdq.clear();
  s.completions.clear();
  s.outstanding = 0;
  s.prefill = true;
  s.now = 0;
  s.next_bank_pre = 0;
  s.transactions = 0;
  s.start_dram_cycle = m_dram_cycle;
  s.start_stats = m_pim->stats();
  if (m_config->m_pim_config.align_refresh) refresh_reset();
  if (m_pim_trace_file)
    fprintf(m_pim_trace_file, "# %s channel %u start DRAM cycle %llu\n",
            s.prog->name.c_str(), id, m_dram_cycle);
}

// One DRAM cycle of the PIM controller, in the order of PIMSimulator's
// MemorySystem::update() and MemoryController::update()
void dram_t::pim_cycle() {
  pim_session &s = *m_pim_sess;
  const pim_config &cfg = m_config->m_pim_config;

  // A channel that is idle at a drain point (or that just started) takes the
  // next batch of transactions at once, like transactions added between two
  // PIMBench run() calls. Otherwise one pending transaction enters the
  // transaction queue per cycle.
  if (s.cursor.at_fence() && !s.outstanding) {
    s.cursor.pass_fence();
    s.prefill = true;
  }
  if (s.prefill) {
    while (s.transq.size() < cfg.trans_queue_size && s.cursor.can_release()) {
      s.transq.push_back(s.cursor.next());
      s.outstanding++;
      s.transactions++;
    }
    s.prefill = false;
  } else if (s.transq.size() < cfg.trans_queue_size && s.cursor.can_release()) {
    s.transq.push_back(s.cursor.next());
    s.outstanding++;
    s.transactions++;
  }

  // CommandQueue::pop(): at most one command on the command bus
  bool issued =
      pim_issue_refresh() || pim_issue_command() || pim_issue_precharge();
  if (!issued && !s.cmdq.empty()) m_pim->stats().blocked_cycles++;

  // MemoryController::updateTransactionQueue(): one transaction becomes a
  // command per cycle
  if (!s.transq.empty() && s.cmdq.size() < cfg.cmd_queue_size) {
    s.cmdq.push_back(s.transq.front());
    s.transq.erase(s.transq.begin());
  }

  m_pim->stats().mode_cycles[(int)m_pim->mode()]++;

  // transactions whose data transfer completes this cycle
  while (!s.completions.empty() && s.completions.front() <= s.now) {
    std::pop_heap(s.completions.begin(), s.completions.end(),
                  std::greater<unsigned long long>());
    s.completions.pop_back();
    s.outstanding--;
  }

  if (s.cursor.done() && !s.outstanding) {
    // PIMBench counts this cycle as the last one of the kernel
    s.phase = pim_session::IDLE;
    s.end_dram_cycle = s.start_dram_cycle + s.now;
    s.kernel_stats = m_pim->stats() - s.start_stats;
    s.finished = true;
    if (m_pim_trace_file) {
      fprintf(m_pim_trace_file, "# end after %llu DRAM cycles\n", s.now + 1);
      fflush(m_pim_trace_file);
    }
  }
  s.now++;
}

bool dram_t::pim_bank_idle(unsigned b) const {
  return bk[b]->state == BANK_IDLE && !bk[b]->RPc && !RFCc;
}

bool dram_t::pim_act_issuable(unsigned b) {
  unsigned grp = get_bankgrp_number(b);
  return m_pim->act_allowed(b) && !bk[b]->RCc && !bk[b]->RPc && !RRDc &&
         !bkgrp[grp]->RRDLc && !RFCc;
}

bool dram_t::pim_pre_issuable(unsigned b) {
  unsigned grp = get_bankgrp_number(b);
  return bk[b]->state == BANK_ACTIVE && !bk[b]->RASc && !bk[b]->WTPc &&
         !bk[b]->RTPc && !bkgrp[grp]->RTPLc;
}

bool dram_t::pim_col_issuable(const pim_req_t &r) {
  bank_t *b = bk[r.bank];
  unsigned grp = get_bankgrp_number(r.bank);
  if (b->state != BANK_ACTIVE || b->curr_row != r.row || CCDc ||
      bkgrp[grp]->CCDLc)
    return false;
  return r.write ? (!b->RCDWRc && !RTWc) : (!b->RCDc && !WTRc);
}

// CommandQueue::process_refresh(): precharge open banks of the rank, then
// refresh it once no bank is active; otherwise fall through to the other
// commands (activates are not held off)
bool dram_t::pim_issue_refresh() {
  if (!refresh_pending) return false;
  if (refresh_pending_rank == 0) {
    bool active = false;
    for (unsigned b = 0; b < m_config->nbk; b++) {
      if (bk[b]->state == BANK_ACTIVE) {
        active = true;
        if (pim_pre_issuable(b)) {
          pim_issue_pre(b);
          return true;
        }
      }
    }
    if (active) return false;
    RFCc = m_config->tRFC;
    m_pim->stats().n_ref++;
  } else {
    m_pim->stats().n_ref_other_rank++;
  }
  if (m_pim_trace_file)
    fprintf(m_pim_trace_file, "%llu REF ra%u\n", m_pim_sess->now,
            refresh_pending_rank);
  refresh_pending = false;
  m_pim_sess->cmds_issued++;
  return true;
}

// CommandQueue::process_command(): the oldest issuable read or write that
// is not held back by a barrier or an older access to the same address,
// otherwise an activate for the oldest command whose bank is idle (looking
// past a barrier at the head of the queue, up to the next barrier)
bool dram_t::pim_issue_command() {
  std::vector<pim_req_t> &q = m_pim_sess->cmdq;
  for (size_t i = 0; i < q.size(); i++) {
    if (!pim_col_issuable(q[i])) continue;
    if (i != 0 && q[i].barrier) break;
    bool depend = false;
    for (size_t j = 0; j < i; j++) {
      if ((q[i].bank == q[j].bank && q[i].row == q[j].row &&
           q[i].col == q[j].col) ||
          q[j].barrier) {
        depend = true;
        break;
      }
    }
    if (!depend) {
      pim_req_t r = q[i];
      q.erase(q.begin() + i);
      pim_issue_col(r);
      return true;
    }
  }
  for (size_t i = 0; i < q.size(); i++) {
    if (i != 0 && q[i].barrier) break;
    if (pim_bank_idle(q[i].bank) && pim_act_issuable(q[i].bank)) {
      pim_issue_act(q[i]);
      return true;
    }
  }
  return false;
}

// CommandQueue::process_precharge(): round robin over the banks, close an
// open row that no command up to the first barrier needs
bool dram_t::pim_issue_precharge() {
  pim_session &s = *m_pim_sess;
  unsigned start = s.next_bank_pre;
  do {
    unsigned b = s.next_bank_pre;
    bool found = false;
    for (const pim_req_t &p : s.cmdq) {
      if (p.bank == b && bk[b]->state == BANK_ACTIVE &&
          p.row == bk[b]->curr_row)
        found = true;
      if (p.barrier) break;
    }
    if (!found && pim_pre_issuable(b)) {
      pim_issue_pre(b);
      return true;
    }
    s.next_bank_pre = (s.next_bank_pre + 1) % m_config->nbk;
  } while (s.next_bank_pre != start);
  return false;
}

void dram_t::pim_issue_act(const pim_req_t &r) {
  unsigned b = r.bank;
  unsigned grp = get_bankgrp_number(b);
  pim_trace("ACT", b, r.row, r.col, r.kind);
  bk[b]->curr_row = r.row;
  bk[b]->state = BANK_ACTIVE;
  RRDc = m_config->tRRD;
  bkgrp[grp]->RRDLc = m_config->tRRDL;
  bk[b]->RCDc = m_config->tRCD;
  bk[b]->RCDWRc = m_config->tRCDWR;
  bk[b]->RASc = m_config->tRAS;
  bk[b]->RCc = m_config->tRC;

  pim_stats &st = m_pim->stats();
  st.n_act++;
  if (m_pim->mode() != pim_mode::SB) st.n_allbank_act++;
  m_pim->on_activate(b, r.row, r.col);
  m_pim_sess->cmds_issued++;
}

void dram_t::pim_issue_pre(unsigned b) {
  unsigned row = bk[b]->curr_row;
  pim_trace("PRE", b, row, 0, -1);
  bk[b]->state = BANK_IDLE;
  bk[b]->RPc = m_config->tRP;

  pim_stats &st = m_pim->stats();
  st.n_pre++;
  if (m_pim->mode() != pim_mode::SB) st.n_allbank_pre++;
  m_pim->on_precharge(b, row);
  m_pim_sess->cmds_issued++;
}

void dram_t::pim_issue_col(const pim_req_t &r) {
  pim_session &s = *m_pim_sess;
  unsigned grp = get_bankgrp_number(r.bank);
  pim_trace(r.write ? "WR" : "RD", r.bank, r.row, r.col, r.kind);
  const unsigned data_cycles = m_config->BL / m_config->data_command_freq_ratio;
  unsigned long long done;
  CCDc = m_config->tCCD;
  bkgrp[grp]->CCDLc = m_config->tCCDL;
  if (r.write) {
    WTRc = m_config->tWTR;
    bk[r.bank]->WTPc = m_config->tWTP;
    // write data follows WL cycles after the command
    done = s.now + m_config->WL + data_cycles;
  } else {
    RTWc = m_config->tRTW;
    bk[r.bank]->RTPc = m_config->tRTP;
    bkgrp[grp]->RTPLc = m_config->tRTPL;
    // the command reaches the device after tCMD, data returns after CL
    done =
        s.now + m_config->m_pim_config.cmd_latency + m_config->CL + data_cycles;
  }
  s.completions.push_back(done);
  std::push_heap(s.completions.begin(), s.completions.end(),
                 std::greater<unsigned long long>());

  pim_stats &st = m_pim->stats();
  bool all_bank = m_pim->mode() != pim_mode::SB;
  if (r.write) {
    st.n_wr++;
    if (all_bank) st.n_allbank_wr++;
  } else {
    st.n_rd++;
    if (all_bank) st.n_allbank_rd++;
  }
  m_pim->on_column(r);
  s.cmds_issued++;
}

void dram_t::pim_trace(const char *cmd, unsigned bank, unsigned row,
                       unsigned col, int kind) {
  if (!m_pim_trace_file) return;
  fprintf(m_pim_trace_file, "%llu %s ba%u ro%u co%u %s %s\n", m_pim_sess->now,
          cmd, bank, row, col, pim_mode_str(m_pim->mode()),
          kind >= 0 ? pim_req_kind_str(kind) : "-");
}

void dram_t::pim_print(FILE *fp) const {
  const pim_stats &st = m_pim->stats();
  fprintf(fp,
          "PIM[%d]: mode=%s cycles_sb=%llu cycles_hab=%llu cycles_hab_pim=%llu "
          "n_act=%llu n_pre=%llu n_rd=%llu n_wr=%llu n_ref=%llu "
          "allbank_act=%llu allbank_pre=%llu allbank_rd=%llu allbank_wr=%llu "
          "pim_cmds=%llu mac_lane_ops=%llu\n",
          id, pim_mode_str(m_pim->mode()), st.mode_cycles[0], st.mode_cycles[1],
          st.mode_cycles[2], st.n_act, st.n_pre, st.n_rd, st.n_wr, st.n_ref,
          st.n_allbank_act, st.n_allbank_pre, st.n_allbank_rd, st.n_allbank_wr,
          st.n_pim_cmds, st.mac_lane_ops);
}
