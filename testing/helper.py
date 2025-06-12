import  xarray as xr

ds = xr.open_dataset('../era_test/sagnn_2020.nc')

# Split the file on months 1 to 10 and 11 to 12

ds_jan_oct = ds.sel(time=slice('2020-01-01','2020-10-31T23:00:00'))

ds_nov_dec = ds.sel(time=slice('2020-11-01','2020-12-31T23:00:00'))

print(f'Shapes: {ds_jan_oct.u.values.shape}, {ds_nov_dec.u.values.shape}')

# Save to two different .nc files

ds_jan_oct.to_netcdf('era_jan_oct_og.nc')
ds_nov_dec.to_netcdf('era_nov_dec.nc')