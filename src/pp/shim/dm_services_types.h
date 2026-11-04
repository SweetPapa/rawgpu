/* shim for display/dc/dm_services_types.h: the watermark types
 * smu7_set_watermarks_for_clocks_ranges() takes (copied from Linux 05441d697f7f). */
#include "rg_pp_kcompat.h"
#ifndef RG_PP_DM_SERVICES_TYPES_H
#define RG_PP_DM_SERVICES_TYPES_H
#define MAX_WM_SETS 4
enum dm_pp_wm_set_id {
  WM_SET_A = 0,
  WM_SET_B,
  WM_SET_C,
  WM_SET_D,
  WM_SET_INVALID = 0xffff,
};
struct dm_pp_clock_range_for_wm_set {
  enum dm_pp_wm_set_id wm_set_id;
  uint32_t wm_min_eng_clk_in_khz;
  uint32_t wm_max_eng_clk_in_khz;
  uint32_t wm_min_mem_clk_in_khz;
  uint32_t wm_max_mem_clk_in_khz;
};
struct dm_pp_wm_sets_with_clock_ranges {
  uint32_t num_wm_sets;
  struct dm_pp_clock_range_for_wm_set wm_clk_ranges[MAX_WM_SETS];
};
struct dm_pp_clock_range_for_dmif_wm_set_soc15 {
  enum dm_pp_wm_set_id wm_set_id;
  uint32_t wm_min_dcfclk_clk_in_khz;
  uint32_t wm_max_dcfclk_clk_in_khz;
  uint32_t wm_min_mem_clk_in_khz;
  uint32_t wm_max_mem_clk_in_khz;
};
struct dm_pp_clock_range_for_mcif_wm_set_soc15 {
  enum dm_pp_wm_set_id wm_set_id;
  uint32_t wm_min_socclk_clk_in_khz;
  uint32_t wm_max_socclk_clk_in_khz;
  uint32_t wm_min_mem_clk_in_khz;
  uint32_t wm_max_mem_clk_in_khz;
};
struct dm_pp_wm_sets_with_clock_ranges_soc15 {
  uint32_t num_wm_dmif_sets;
  uint32_t num_wm_mcif_sets;
  struct dm_pp_clock_range_for_dmif_wm_set_soc15 wm_dmif_clocks_ranges[MAX_WM_SETS];
  struct dm_pp_clock_range_for_mcif_wm_set_soc15 wm_mcif_clocks_ranges[MAX_WM_SETS];
};
#endif
