// HBM-PIM (Samsung HBM2 processing-in-memory) configuration.
//
// The model is a timing-only port of Samsung's PIMSimulator (a DRAMSim2
// fork, equivalent to its NO_STORAGE build): every DRAM
// channel holds one PIM rank whose mode (SB / HAB / HAB_PIM) is switched by
// accesses to reserved rows, and whose PIM blocks step through a programmable
// command register file (CRF) on every column command issued in PIM mode.

#ifndef PIM_CONFIG_H
#define PIM_CONFIG_H

class OptionParser;
class memory_config;

class pim_config {
 public:
  pim_config() : enabled(false), cmd_trace(nullptr) {}

  void reg_options(OptionParser *opp);
  void init(const memory_config *mem_config);

  bool is_reserved_row(unsigned row) const {
    return (row >> reserved_row_bit) & 1;
  }

  bool enabled;

  // PIM block geometry
  unsigned num_blocks;       // PIM blocks per channel
  unsigned banks_per_block;  // banks sharing one PIM block (even/odd pair)
  unsigned crf_size;         // CRF entries (32-bit instructions)
  unsigned grf_size;         // entries per GRF bank (GRF_A and GRF_B)
  unsigned cols_per_row;     // 32 B columns per DRAM row

  // reserved rows (row address bit reserved_row_bit set)
  unsigned reg_row;   // PIM register row (control, SRF, CRF, GRF)
  unsigned abmr_row;  // all-bank mode register (SB -> HAB)
  unsigned sbmr_row;  // single-bank mode register (HAB -> SB)
  unsigned park_row;  // dummy row used by park-in / park-out
  unsigned reserved_row_bit;

  // in-order PIM controller (DRAMSim2 transaction / command queues)
  unsigned trans_queue_size;
  unsigned cmd_queue_size;
  unsigned cmd_latency;  // tCMD: command to device latency

  bool selfcheck;      // check closed-form command counts per PIM kernel
  bool align_refresh;  // restart refresh countdowns at PIM kernel launch

  char *cmd_trace;        // file to write the PIM command trace to
  unsigned cmd_trace_ch;  // channel whose commands are traced
};

#endif
