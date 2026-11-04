/* Symbols the vendored powerplay files reference for other chips.
 *
 * hwmgr.c's hwmgr_early_init() switches on the chip and names every family's
 * function tables; smu7_baco.c names the Tonga and Fiji BACO paths. rawgpu
 * vendors only the Polaris 10 path (docs/DPM.md), so the rest are empty here
 * and fail loudly if anything reaches them. The v0 PowerPlay parser
 * (processpptables.c) is for chips before Tonga; Polaris uses v1.
 */
#include "hwmgr.h"
#include "processpptables.h"
#include "smumgr.h"
#include "tonga_baco.h"
#include "fiji_baco.h"
#include "vega10_hwmgr.h"

const struct pp_smumgr_func ci_smu_funcs, smu8_smu_funcs, iceland_smu_funcs, tonga_smu_funcs, fiji_smu_funcs,
    vegam_smu_funcs, vega10_smu_funcs, vega12_smu_funcs, smu10_smu_funcs, vega20_smu_funcs;
const struct pp_table_func pptable_funcs;

static int not_polaris(const char *what) {
  pr_err("%s: rawgpu only carries amdgpu's Polaris 10 powerplay path\n", what);
  return -EINVAL;
}

int smu8_init_function_pointers(struct pp_hwmgr *hwmgr) { return not_polaris(__func__); }
int smu10_init_function_pointers(struct pp_hwmgr *hwmgr) { return not_polaris(__func__); }
int vega10_hwmgr_init(struct pp_hwmgr *hwmgr) { return not_polaris(__func__); }
int vega12_hwmgr_init(struct pp_hwmgr *hwmgr) { return not_polaris(__func__); }
int vega20_hwmgr_init(struct pp_hwmgr *hwmgr) { return not_polaris(__func__); }
int tonga_baco_set_state(struct pp_hwmgr *hwmgr, enum BACO_STATE state) { return not_polaris(__func__); }
int fiji_baco_set_state(struct pp_hwmgr *hwmgr, enum BACO_STATE state) { return not_polaris(__func__); }

int pp_tables_get_num_of_entries(struct pp_hwmgr *hwmgr, unsigned long *num_of_entries) {
  return not_polaris(__func__);
}
int pp_tables_get_entry(struct pp_hwmgr *hwmgr, unsigned long entry_index, struct pp_power_state *ps,
                        pp_tables_hw_clock_info_callback func) {
  return not_polaris(__func__);
}
