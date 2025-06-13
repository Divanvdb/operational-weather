
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import xarray as xr
import cartopy.crs as ccrs



def plot_example(
    prediction,
    target,
    offset: int = 0,
    frame_rate: int = 2,
    levels: int = 20,
    save_path: str = "wind_speed_animation.mp4"  # Save path here
):
    lon, lat = target.longitude, target.latitude
    bounds = [lon.min().item(), lon.max().item(), lat.min().item(), lat.max().item()]

    vmin = min(prediction.values.min(), target.values.min())
    vmax = max(prediction.values.max(), target.values.max())

    init_time_pred = prediction.time.values
    init_time_target = target.time.values

    times_pred = np.array(prediction["prediction_timedelta"].values).astype("timedelta64[3h]")
    times_pred = np.array([np.datetime64(init_time_pred + time) for time in times_pred])

    times_target = np.array(target["prediction_timedelta"].values).astype("timedelta64[3h]")
    times_target = np.array([np.datetime64(init_time_target + time) for time in times_target])

    times = times_target

    fig, axs = plt.subplots(1, 3, figsize=(14, 7), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)

    for ax in axs:
        ax.coastlines()
        ax.set_extent(bounds, crs=ccrs.PlateCarree())

    pred_states = prediction.transpose("prediction_timedelta", "latitude", "longitude").values
    target_states = target.transpose("prediction_timedelta", "latitude", "longitude").values
    err_states = np.abs(target_states - pred_states)
    emin, emax = err_states.min(), err_states.max()

    def animate(i):
        for ax in axs:
            ax.clear()
            ax.coastlines()
            ax.set_extent(bounds, crs=ccrs.PlateCarree())

        axs[0].contourf(lon, lat, pred_states[i], levels=levels, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        axs[1].contourf(lon, lat, target_states[i], levels=levels, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        axs[2].contourf(lon, lat, err_states[i], levels=levels, vmin=emin, vmax=emax, transform=ccrs.PlateCarree(), cmap="coolwarm")

        axs[0].set_title(f"Predicted {i} - {times[i]}")
        axs[1].set_title(f"Actual {i} - {times[i]}")
        axs[2].set_title(f"Error {i} - {times[i]}")

    frames = pred_states.shape[0]
    interval = 1000 / frame_rate

    ani = FuncAnimation(fig, animate, frames=frames, interval=interval)

    # Save as MP4 or GIF
    ani.save("animation.gif", writer='pillow', fps=frame_rate) # Requires ffmpeg installed
    # ani.save("wind_speed_animation.gif", writer='pillow', fps=frame_rate)  # GIF alternative

    plt.close(fig)
    print(f"Animation saved to {save_path}")


if __name__ == "__main__":
    ds = xr.open_dataset('forecasted_weather.nc')
    pred = ds["wind_speed"]
    actual = ds["target"]

    print(ds)
    plot_example(pred, actual, offset=0, frame_rate=2, levels=20)