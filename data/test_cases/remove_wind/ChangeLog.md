### Previous Version
updated_econ_metrics

### Changes
- Moves from Hybrid system to PV Batt system to remove wind
- Increases PV size from 230MW to 320MW
    - To accomodate the loss of 61MW of wind generation, we needed at least 290MW of PV (assuming coincident generation)
    - Generation is not perfectly coincident, so we need to increase the PV by a bit more and increase the battery by more as well
- Increases minimum battery SOC from 10% to 15%
    - For reliability
- Increases battery size from 90MW/360MWh to 120MW/480MWh
    - Since we've just changed our usable DOD from 85% to 80%, we need to increase the battery size by at least 6.25% (85/80 = 1.0625). This would yield a new battery size of 96MW.
    - After running the battery dispatch model, 96MW proved not to be large enough (we had some instances of violating the geothermal ramp rates and failing to meet the load), so we needed to increase this even further.
- Allows battery to charge from grid to enable the battery to charge from geothermal. The charge rate is still limited in battery_dispatch_model, so we shouldn't be charging with energy that we don't actually have available
- Also allows battery to charge from clipped DC power, which gets more utilization from the PV 

    