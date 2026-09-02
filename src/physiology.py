from src.config import GRID_MINUTES, INSULIN_DIA_MIN, MEAL_CARB_DIA_MIN, CORRECTION_CARB_DIA_MIN

def remaining_fraction(minutes, duration):
    x = minutes / duration
    # x = 0 if x < 0, x = 1 if x > 1
    x = min(max(x, 0), 1)

    absorbed = 3*x**2 - 2*x**3

    return 1 - absorbed
    

def decay_sum(amounts, duration):
    steps = duration // GRID_MINUTES

    # running total
    total = amounts * 0

    for i in range(steps):
        minutes = i * GRID_MINUTES
        weight = remaining_fraction(minutes, duration)

        # a dose from i slots ago still contributes 'weight' of itself 
        total = total + amounts.shift(i).fillna(0) * weight

    return total

def add_iob_cob(df):
    df_copy = df.copy()

    df_copy['iob'] = decay_sum(df_copy['bolus_insulin'], INSULIN_DIA_MIN)
    df_copy['meal_cob'] = decay_sum(df_copy['meal_carbs'], MEAL_CARB_DIA_MIN)
    df_copy['correction_cob'] = decay_sum(df_copy['correction_carbs'], CORRECTION_CARB_DIA_MIN)

    df_copy['total_cob'] = df_copy['meal_cob'] + df_copy['correction_cob']

    return df_copy
    