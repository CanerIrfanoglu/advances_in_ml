import numpy as np

# --- Power Function Bet Sizing (Exercise 10.6 & 10.7) ---

def betSize_power(x_divergence, k_calibrated, p_exponent):
    """
    Power function based bet size.
    x_divergence: f - mP
    k_calibrated: scaling factor (divergence at which bet is full if p=1)
    p_exponent: power
    Returns fractional bet size between -1 and 1.
    """
    if k_calibrated == 0: # Avoid division by zero if k is not set or invalid
        return 0.0
    
    # Normalized divergence
    norm_x = abs(x_divergence) / k_calibrated
    
    # Apply power and cap
    size_abs = min(1.0, norm_x**p_exponent)
    
    return np.sign(x_divergence) * size_abs

def getTPos_power(k_calibrated, p_exponent, f_forecast, mP_marketPrice, maxPos_qty):
    """Target position using power function bet sizing."""
    x_divergence = f_forecast - mP_marketPrice
    fractional_bet = betSize_power(x_divergence, k_calibrated, p_exponent)
    return int(fractional_bet * maxPos_qty)

def invPrice_power(f_forecast, k_calibrated, p_exponent, m_fractional_bet):
    """
    Inverse price calculation for power function.
    Calculates market price mP given f, k, p, and fractional bet m.
    m_fractional_bet is the desired fractional position size (e.g., j / maxPos).
    """
    if m_fractional_bet == 0:
        # If desired fractional bet is 0, then divergence must be 0, so mP = f
        return f_forecast
    
    # Calculate |x| = k * |m|^(1/p)
    abs_x_divergence = k_calibrated * (abs(m_fractional_bet)**(1.0 / p_exponent))
    
    # x = sign(m) * |x|
    signed_x_divergence = np.sign(m_fractional_bet) * abs_x_divergence
    
    # mP = f - x
    return f_forecast - signed_x_divergence

def limitPrice_power(tPos_target, pos_current, f_forecast, k_calibrated, p_exponent, maxPos_qty):
    """Limit price calculation using power function bet sizing."""
    if tPos_target == pos_current:
        # If current position is already the target position,
        # technically no order needed. Limit price could be current market price
        # or forecast, or not well-defined. For now, let's return forecast.
        # Or better, this case should be handled by not placing an order.
        # If forced to return a price, it's tricky. Let's assume an order *is* placed.
        print("Warning: tPos == pos in limitPrice_power. Order size is 0.")
        return f_forecast # Or some other appropriate default or error

    order_size = tPos_target - pos_current
    sgn = np.sign(order_size) # 1 if buying more, -1 if selling more
    
    limit_price_sum = 0
    
    # Sum over the units being added/removed to reach tPos
    # j represents the absolute size of the position for that unit
    # Example: pos=0, tPos=3. sgn=1. j iterates for units that make pos 1, 2, 3.
    # Example: pos=5, tPos=2. sgn=-1. j iterates for units that make pos 4, 3, 2.
    
    # Iterate over the "slots" in the position from current to target
    # The j values for invPrice should represent the target fractional size
    # *after* that j-th unit of the order is hypothetically filled.
    
    num_steps_in_order = abs(order_size)
    if num_steps_in_order == 0: return f_forecast # Should be caught by tPos == pos_current

    current_abs_pos_j = abs(pos_current)

    for i in range(1, num_steps_in_order + 1):
        # The target absolute position level after this i-th part of the order
        target_j_level_abs = abs(pos_current + sgn * i)
        
        # Fractional bet size for this level
        m_j_fractional = np.sign(pos_current + sgn * i) * (target_j_level_abs / float(maxPos_qty))
        
        # We need to ensure m_j_fractional doesn't exceed +/-1 for invPrice,
        # though target_j_level_abs should not exceed maxPos_qty if tPos is capped.
        m_j_fractional = np.clip(m_j_fractional, -1.0, 1.0)

        if abs(m_j_fractional) > 1.0 : # Should not happen if tPos is capped correctly
            print(f"Warning: m_j_fractional {m_j_fractional} out of bounds for j_level {target_j_level_abs}")

        price_for_jth_unit = invPrice_power(f_forecast, k_calibrated, p_exponent, m_j_fractional)
        limit_price_sum += price_for_jth_unit
        
    return limit_price_sum / num_steps_in_order


def getK_power(x_star_divergence, m_star_fractional_bet, p_exponent):
    """Calibrates k for the power function given a target m* at x* for a fixed p."""
    if m_star_fractional_bet == 0:
        return np.inf # k would need to be infinite if m* is 0 for non-zero x*
    if p_exponent == 0: # Avoid division by zero or 0^0 issues
        raise ValueError("p_exponent cannot be zero for calibration.")
    if abs(m_star_fractional_bet) >= 1.0 and x_star_divergence > 0 : # if m_star is +/-1
        return abs(x_star_divergence) # k is the divergence at which full bet is reached
        
    return abs(x_star_divergence) / (abs(m_star_fractional_bet)**(1.0 / p_exponent))

# --- Modified main for Power Function (Exercise 10.7) ---
def main_power():
    pos, maxPos, mP, f = 0, 100, 100, 115
    
    # Parameters for power function calibration
    # Let's choose an exponent, e.g., p=0.5 (square root like curve, aggressive for small x)
    # or p=1 (linear up to k), or p=2 (quadratic, less aggressive for small x)
    p_exponent_val = 1.0 # Try p=1 for linear sizing up to k
    
    # Calibration parameters for k:
    # "When divergence is $10, I want my fractional bet size to be 0.95."
    calibrate_divergence = 10.0
    calibrate_m_star = 0.95
    
    k = getK_power(calibrate_divergence, calibrate_m_star, p_exponent_val)
    print(f"Calibrated k for p={p_exponent_val}: {k:.4f}")
    
    tPos = getTPos_power(k, p_exponent_val, f, mP, maxPos)
    print(f"Target Position (Power func): {tPos}")
    
    if tPos == pos:
        print("No order needed, current position is target position.")
        lP = mP # Or f, or undefined.
    else:
        lP = limitPrice_power(tPos, pos, f, k, p_exponent_val, maxPos)
    
    print(f"Limit Price (Power func, p={p_exponent_val}): {lP:.4f}")

    # Example with p=0.5
    p_exponent_val = 0.5
    k = getK_power(calibrate_divergence, calibrate_m_star, p_exponent_val)
    print(f"\nCalibrated k for p={p_exponent_val}: {k:.4f}")
    tPos = getTPos_power(k, p_exponent_val, f, mP, maxPos)
    print(f"Target Position (Power func): {tPos}")
    if tPos == pos:
        print("No order needed, current position is target position.")
        lP = mP
    else:
        lP = limitPrice_power(tPos, pos, f, k, p_exponent_val, maxPos)
    print(f"Limit Price (Power func, p={p_exponent_val}): {lP:.4f}")

    return

if __name__ == '__main__':
    # main() # Original sigmoid
    main_power() # New power function