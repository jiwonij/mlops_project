def prepare_data(train_source, building_info_source):
    if isinstance(train_source, pd.DataFrame):
        train = train_source.copy()
    else:
        train = pd.read_csv(train_source)

    if isinstance(building_info_source, pd.DataFrame):
        building_info = building_info_source.copy()
    else:
        building_info = pd.read_csv(building_info_source)

    train = train.rename(columns={
        "건물번호": "building_number",
        "일시": "date_time",
        "기온(°C)": "temperature",
        "강수량(mm)": "rainfall",
        "풍속(m/s)": "windspeed",
        "습도(%)": "humidity",
        "일조(hr)": "sunshine",
        "일사(MJ/m2)": "solar_radiation",
        "전력소비량(kWh)": "power_consumption",
    })

    if "num_date_time" in train.columns:
        train.drop("num_date_time", axis=1, inplace=True)

    building_info = building_info.rename(columns={
        "건물번호": "building_number",
        "건물유형": "building_type",
        "연면적(m2)": "total_area",
        "냉방면적(m2)": "cooling_area",
        "태양광용량(kW)": "solar_power_capacity",
        "ESS저장용량(kWh)": "ess_capacity",
        "PCS용량(kW)": "pcs_capacity",
    })

    translation_dict = {
        "건물기타": "Other Buildings",
        "공공": "Public",
        "학교": "University",
        "백화점": "Department Store",
        "병원": "Hospital",
        "상용": "Commercial",
        "아파트": "Apartment",
        "연구소": "Research Institute",
        "IDC(전화국)": "IDC",
        "호텔": "Hotel",
    }

    building_info["building_type"] = building_info["building_type"].replace(translation_dict)

    building_info["solar_power_utility"] = np.where(
        building_info["solar_power_capacity"] != "-", 1, 0
    )
    building_info["ess_utility"] = np.where(
        building_info["ess_capacity"] != "-", 1, 0
    )

    df = pd.merge(train, building_info, on="building_number", how="left")

    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values(["building_number", "date_time"]).reset_index(drop=True)

    df["hour"] = df["date_time"].dt.hour
    df["dayofweek"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["diff_1"] = df.groupby("building_number")["power_consumption"].diff(1)
    df["diff_24"] = df.groupby("building_number")["power_consumption"].diff(24)

    df["lag_1"] = df.groupby("building_number")["power_consumption"].shift(1)
    df["lag_24"] = df.groupby("building_number")["power_consumption"].shift(24)
    df["lag_168"] = df.groupby("building_number")["power_consumption"].shift(168)

    df["roll_mean_6"] = (
        df.groupby("building_number")["power_consumption"]
        .shift(1)
        .rolling(6)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["roll_std_6"] = (
        df.groupby("building_number")["power_consumption"]
        .shift(1)
        .rolling(6)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["roll_mean_24"] = (
        df.groupby("building_number")["power_consumption"]
        .shift(1)
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["roll_std_24"] = (
        df.groupby("building_number")["power_consumption"]
        .shift(1)
        .rolling(24)
        .std()
        .reset_index(level=0, drop=True)
    )

    df = df.dropna().reset_index(drop=True)

    return df