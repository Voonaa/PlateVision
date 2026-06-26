# Konfigurasi Preset Adaptive Preprocessing untuk PlateVision

PRESETS = {
    "Bright Environment": {
        "d": 9,
        "sigma_color": 15,
        "sigma_space": 15,
        "canny_low": 50,
        "canny_high": 220,
        "approx_epsilon": 10
    },
    "Low Light": {
        "d": 11,
        "sigma_color": 25,
        "sigma_space": 25,
        "canny_low": 20,
        "canny_high": 120,
        "approx_epsilon": 10
    },
    "Night": {
        "d": 13,
        "sigma_color": 35,
        "sigma_space": 35,
        "canny_low": 15,
        "canny_high": 100,
        "approx_epsilon": 10
    },
    "Rain/Fog": {
        "d": 15,
        "sigma_color": 50,
        "sigma_space": 50,
        "canny_low": 15,
        "canny_high": 100,
        "approx_epsilon": 10
    },
    "Motion Blur": {
        "d": 5,
        "sigma_color": 10,
        "sigma_space": 10,
        "canny_low": 40,
        "canny_high": 180,
        "approx_epsilon": 10
    },
    "Normal": {
        "d": 11,
        "sigma_color": 17,
        "sigma_space": 17,
        "canny_low": 30,
        "canny_high": 200,
        "approx_epsilon": 10
    }
}

# Mendapatkan parameter pre-processing berdasarkan mode terpilih
def get_adaptive_parameters(mode, metrics=None):
    auto_chosen_name = "Normal"
    
    if mode == "Auto (Recommended)":
        if metrics:
            brightness = metrics.get("brightness", 120)
            sharpness = metrics.get("sharpness", 200)
            noise = metrics.get("noise", 15)
            
            # Logika keputusan cerdas untuk pemilihan preset
            if brightness < 45 and noise > 25:
                auto_chosen_name = "Night"
            elif brightness < 65:
                auto_chosen_name = "Low Light"
            elif brightness > 175:
                auto_chosen_name = "Bright Environment"
            elif noise > 32:
                auto_chosen_name = "Rain/Fog"
            elif sharpness < 90:
                auto_chosen_name = "Motion Blur"
            else:
                auto_chosen_name = "Normal"
        else:
            auto_chosen_name = "Normal"
            
        params = PRESETS[auto_chosen_name].copy()
        params["mode_applied"] = f"Auto ➔ {auto_chosen_name}"
        return params
        
    elif mode in PRESETS:
        params = PRESETS[mode].copy()
        params["mode_applied"] = mode
        return params
        
    else:
        # Fallback jika mode Manual atau lainnya
        return {
            "d": 11,
            "sigma_color": 17,
            "sigma_space": 17,
            "canny_low": 30,
            "canny_high": 200,
            "approx_epsilon": 10,
            "mode_applied": "Manual"
        }
