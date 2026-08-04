"""Generate a multi-sine noise signal and display it interactively.

Default behavior matches the requested setup:
- 100 sine waves
- frequencies evenly spaced from 0.1 Hz to 6.0 Hz
- random phase for each wave
- amplitudes drawn from a Gaussian centered at amplitude_constant / 0.6366
- 200 second duration sampled at 100 Hz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def generate_multisine_noise(
    num_waves: int = 100,
    min_frequency_hz: float = 0.1,
    max_frequency_hz: float = 6.0,
    duration_s: float = 200.0,
    sample_rate_hz: float = 100.0,
    num_samples: Optional[int] = None,
    amplitude_constant: float = 1.0,
    amplitude_std: Optional[float] = None,
    target_std: Optional[float] = None,
    target_mae: Optional[float] = None,
    seed: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Create a summed multi-sine signal with random phase and Gaussian amplitudes.

    `amplitude_std` defaults to 10% of the Gaussian mean because the user only
    specified the mean center, not the spread.
    """

    if num_waves <= 0:
        raise ValueError("num_waves must be positive.")
    if min_frequency_hz <= 0 or max_frequency_hz <= 0:
        raise ValueError("Frequencies must be positive.")
    if max_frequency_hz < min_frequency_hz:
        raise ValueError("max_frequency_hz must be >= min_frequency_hz.")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive.")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    if num_samples is not None and num_samples <= 0:
        raise ValueError("num_samples must be positive when provided.")
    if target_std is not None and target_std < 0:
        raise ValueError("target_std must be non-negative when provided.")
    if target_mae is not None and target_mae < 0:
        raise ValueError("target_mae must be non-negative when provided.")
    if target_std is not None and target_mae is not None:
        raise ValueError("Provide only one of target_std or target_mae.")

    rng = np.random.default_rng(seed)

    if num_samples is None:
        time_s = np.arange(0.0, duration_s, 1.0 / sample_rate_hz, dtype=np.float64)
    else:
        time_s = np.arange(num_samples, dtype=np.float64) / sample_rate_hz
    frequencies_hz = np.linspace(min_frequency_hz, max_frequency_hz, num_waves, dtype=np.float64)
    phase_rad = rng.uniform(0.0, 2.0 * np.pi, size=num_waves)

    amplitude_mean = amplitude_constant / 0.6366
    resolved_amplitude_std = amplitude_std if amplitude_std is not None else 0.1 * amplitude_mean
    amplitudes = rng.normal(loc=amplitude_mean, scale=resolved_amplitude_std, size=num_waves)

    wave_bank = amplitudes[:, None] * np.sin(
        2.0 * np.pi * frequencies_hz[:, None] * time_s[None, :] + phase_rad[:, None]
    )
    raw_noise_signal = wave_bank.sum(axis=0)
    centered_noise_signal = raw_noise_signal - np.mean(raw_noise_signal)
    raw_noise_std = float(np.std(centered_noise_signal))
    raw_noise_mae = float(np.mean(np.abs(centered_noise_signal)))

    noise_signal = centered_noise_signal
    scale_factor = 1.0
    if target_mae is not None:
        if raw_noise_mae == 0.0:
            if target_mae == 0.0:
                noise_signal = np.zeros_like(centered_noise_signal)
            else:
                raise RuntimeError("Generated zero-MAE noise; cannot match a positive target MAE.")
        else:
            scale_factor = target_mae / raw_noise_mae
            noise_signal = centered_noise_signal * scale_factor
    elif target_std is not None:
        if raw_noise_std == 0.0:
            if target_std == 0.0:
                noise_signal = np.zeros_like(centered_noise_signal)
            else:
                raise RuntimeError("Generated zero-variance noise; cannot match a positive target std.")
        else:
            scale_factor = target_std / raw_noise_std
            noise_signal = centered_noise_signal * scale_factor

    achieved_std = float(np.std(noise_signal))
    achieved_mae = float(np.mean(np.abs(noise_signal)))

    return {
        "time_s": time_s,
        "noise_signal": noise_signal,
        "raw_noise_signal": raw_noise_signal,
        "frequencies_hz": frequencies_hz,
        "amplitudes": amplitudes,
        "phase_rad": phase_rad,
        "raw_noise_std": np.asarray(raw_noise_std, dtype=np.float64),
        "raw_noise_mae": np.asarray(raw_noise_mae, dtype=np.float64),
        "scale_factor": np.asarray(scale_factor, dtype=np.float64),
        "achieved_std": np.asarray(achieved_std, dtype=np.float64),
        "achieved_mae": np.asarray(achieved_mae, dtype=np.float64),
    }


def display_noise_signal(
    time_s: np.ndarray,
    noise_signal: np.ndarray,
    title: str = "Multi-Sine Noise Signal",
) -> None:
    """Display the synthesized noise signal in an interactive matplotlib window."""

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time_s, noise_signal, linewidth=0.9, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-waves", type=int, default=100, help="Number of sine waves to sum.")
    parser.add_argument("--min-frequency-hz", type=float, default=0.1, help="Minimum sine frequency.")
    parser.add_argument("--max-frequency-hz", type=float, default=6.0, help="Maximum sine frequency.")
    parser.add_argument("--duration-s", type=float, default=200.0, help="Signal duration in seconds.")
    parser.add_argument("--sample-rate-hz", type=float, default=100.0, help="Sampling rate in Hz.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional explicit sample count. When provided, it overrides duration-derived sample count.",
    )
    parser.add_argument(
        "--amplitude-constant",
        type=float,
        default=1.0,
        help="Constant used to set the Gaussian amplitude mean as constant / 0.6366.",
    )
    parser.add_argument(
        "--amplitude-std",
        type=float,
        default=None,
        help="Gaussian amplitude standard deviation. Defaults to 10%% of the mean.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible random phase/amplitude.")
    parser.add_argument(
        "--target-std",
        type=float,
        default=None,
        help="Optional target standard deviation for the final summed noise signal.",
    )
    parser.add_argument(
        "--target-mae",
        type=float,
        default=None,
        help="Optional target mean absolute error for the final summed noise signal.",
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=None,
        help="Optional path to save the generated signal and component parameters as .npz.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Generate the signal without opening the interactive plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = generate_multisine_noise(
        num_waves=args.num_waves,
        min_frequency_hz=args.min_frequency_hz,
        max_frequency_hz=args.max_frequency_hz,
        duration_s=args.duration_s,
        sample_rate_hz=args.sample_rate_hz,
        num_samples=args.num_samples,
        amplitude_constant=args.amplitude_constant,
        amplitude_std=args.amplitude_std,
        target_std=args.target_std,
        target_mae=args.target_mae,
        seed=args.seed,
    )

    if not args.no_show:
        display_noise_signal(
            time_s=result["time_s"],
            noise_signal=result["noise_signal"],
            title=(
                f"Multi-Sine Noise: {args.num_waves} waves, "
                f"{args.min_frequency_hz:.1f}-{args.max_frequency_hz:.1f} Hz"
            ),
        )

    if args.output_data is not None:
        args.output_data.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output_data,
            time_s=result["time_s"],
            noise_signal=result["noise_signal"],
            raw_noise_signal=result["raw_noise_signal"],
            frequencies_hz=result["frequencies_hz"],
            amplitudes=result["amplitudes"],
            phase_rad=result["phase_rad"],
            raw_noise_std=result["raw_noise_std"],
            raw_noise_mae=result["raw_noise_mae"],
            scale_factor=result["scale_factor"],
            achieved_std=result["achieved_std"],
            achieved_mae=result["achieved_mae"],
        )

    print(f"Generated {args.num_waves} sine waves.")
    print(
        "Amplitude Gaussian mean "
        f"= {args.amplitude_constant / 0.6366:.6f}, "
        f"std = {args.amplitude_std if args.amplitude_std is not None else 0.1 * (args.amplitude_constant / 0.6366):.6f}"
    )
    if args.target_mae is not None:
        print(
            f"Target noise MAE = {args.target_mae:.6f}, "
            f"achieved MAE = {float(result['achieved_mae']):.6f}, "
            f"achieved std = {float(result['achieved_std']):.6f}, "
            f"raw MAE = {float(result['raw_noise_mae']):.6f}, "
            f"scale factor = {float(result['scale_factor']):.6f}"
        )
    if args.target_std is not None:
        print(
            f"Target noise std = {args.target_std:.6f}, "
            f"achieved std = {float(result['achieved_std']):.6f}, "
            f"achieved MAE = {float(result['achieved_mae']):.6f}, "
            f"raw std = {float(result['raw_noise_std']):.6f}, "
            f"scale factor = {float(result['scale_factor']):.6f}"
        )
    if args.no_show:
        print("Interactive plot display skipped.")
    else:
        print("Interactive plot displayed.")
    if args.output_data is not None:
        print(f"Saved data to: {args.output_data}")


if __name__ == "__main__":
    main()
