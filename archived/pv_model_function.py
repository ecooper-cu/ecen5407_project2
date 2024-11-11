import pvlib

def model_pv_plant_tracking(data, gcr=0.4):

    backtracking_angles = pvlib.tracking.singleaxis(
        apparent_zenith=data['apparent_zenith'],
        apparent_azimuth=data['azimuth'],
        axis_tilt=0,
        axis_azimuth=180,
        max_angle=90,
        backtrack=True,
        gcr=gcr)
    
    poa = pvlib.irradiance.get_total_irradiance(backtracking_angles['surface_tilt'],
                                            backtracking_angles['surface_azimuth'],
                                            data['zenith'], data['azimuth'],
                                            data['DNI'], data['GHI'], data['DHI'],
                                            data['dni_extra'], model='haydavies')

    aoi = pvlib.irradiance.aoi(backtracking_angles['surface_tilt'],
                                backtracking_angles['surface_azimuth'],
                                data['zenith'], data['azimuth'])
    
    am_rel = pvlib.atmosphere.get_relative_airmass(data['zenith'])
    am_abs = pvlib.atmosphere.get_absolute_airmass(am_rel, data['Pressure'])

    mod_db_cec = pvlib.pvsystem.retrieve_sam('CECMod')
    mod_db_sandia = pvlib.pvsystem.retrieve_sam('SandiaMod')
    module_cec = mod_db_cec['Canadian_Solar_Inc__CS6K_300M']
    module_sandia = mod_db_sandia['Canadian_Solar_CS6X_300M__2013_']

    stc_mod_p = module_cec['STC']
    Gpmp = module_cec['gamma_r']/100
    temp_ref=25.0

    temp_model = 'sapm'
    temp_model_material = 'open_rack_glass_polymer'
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[temp_model][temp_model_material]

    t_cell = pvlib.temperature.sapm_cell(poa['poa_global'],
                                    data['Temperature'].values,
                                    data['Wind Speed'].values,
                                    **temperature_model_parameters)
    eff_irr = pvlib.pvsystem.sapm_effective_irradiance(poa['poa_direct'],
                                                    poa['poa_diffuse'],
                                                    am_abs, aoi, module_sandia)
    
    pvw = pvlib.pvsystem.pvwatts_dc(eff_irr, t_cell, stc_mod_p, Gpmp, temp_ref)
    
    return pvw   


def model_pv_plant_vertical(data, azimuth=270):
    
    poa = pvlib.irradiance.get_total_irradiance(90,
                                                azimuth,
                                                data['zenith'], data['azimuth'],
                                                data['DNI'], data['GHI'], data['DHI'],
                                                data['dni_extra'], model='haydavies')

    aoi = pvlib.irradiance.aoi(90,
                               azimuth,
                               data['zenith'], data['azimuth'])
    
    am_rel = pvlib.atmosphere.get_relative_airmass(data['zenith'])
    am_abs = pvlib.atmosphere.get_absolute_airmass(am_rel, data['Pressure'])

    mod_db_cec = pvlib.pvsystem.retrieve_sam('CECMod')
    mod_db_sandia = pvlib.pvsystem.retrieve_sam('SandiaMod')
    module_cec = mod_db_cec['Canadian_Solar_Inc__CS6K_300M']
    module_sandia = mod_db_sandia['Canadian_Solar_CS6X_300M__2013_']

    stc_mod_p = module_cec['STC']
    Gpmp = module_cec['gamma_r']/100
    temp_ref=25.0

    temp_model = 'sapm'
    temp_model_material = 'open_rack_glass_polymer'
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[temp_model][temp_model_material]

    t_cell = pvlib.temperature.sapm_cell(poa['poa_global'],
                                    data['Temperature'].values,
                                    data['Wind Speed'].values,
                                    **temperature_model_parameters)
    eff_irr = pvlib.pvsystem.sapm_effective_irradiance(poa['poa_direct'],
                                                    poa['poa_diffuse'],
                                                    am_abs, aoi, module_sandia)
    
    pvw = pvlib.pvsystem.pvwatts_dc(eff_irr, t_cell, stc_mod_p, Gpmp, temp_ref)
    
    return pvw                                                                           

def model_pv_plant_fixed_tilt(data, surface_tilt=32, surface_azimuth=180):
    
    poa = pvlib.irradiance.get_total_irradiance(surface_tilt,
                                                surface_azimuth,
                                                data['zenith'], data['azimuth'],
                                                data['DNI'], data['GHI'], data['DHI'],
                                                data['dni_extra'], model='haydavies')

    aoi = pvlib.irradiance.aoi(surface_tilt,
                                surface_azimuth,
                               data['zenith'], data['azimuth'])
    
    am_rel = pvlib.atmosphere.get_relative_airmass(data['zenith'])
    am_abs = pvlib.atmosphere.get_absolute_airmass(am_rel, data['Pressure'])

    mod_db_cec = pvlib.pvsystem.retrieve_sam('CECMod')
    mod_db_sandia = pvlib.pvsystem.retrieve_sam('SandiaMod')
    module_cec = mod_db_cec['Canadian_Solar_Inc__CS6K_300M']
    module_sandia = mod_db_sandia['Canadian_Solar_CS6X_300M__2013_']

    stc_mod_p = module_cec['STC']
    Gpmp = module_cec['gamma_r']/100
    temp_ref=25.0

    temp_model = 'sapm'
    temp_model_material = 'open_rack_glass_polymer'
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[temp_model][temp_model_material]

    t_cell = pvlib.temperature.sapm_cell(poa['poa_global'],
                                    data['Temperature'].values,
                                    data['Wind Speed'].values,
                                    **temperature_model_parameters)
    eff_irr = pvlib.pvsystem.sapm_effective_irradiance(poa['poa_direct'],
                                                    poa['poa_diffuse'],
                                                    am_abs, aoi, module_sandia)
    
    pvw = pvlib.pvsystem.pvwatts_dc(eff_irr, t_cell, stc_mod_p, Gpmp, temp_ref)
    
    return pvw                                                                                                                                                                                                    
