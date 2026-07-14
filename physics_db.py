import sympy as sp

# =========================================================================
# 1. ГЛОБАЛЬНЫЕ СИМВОЛЫ И ПЕРЕМЕННЫЕ (С ЖЕСТКИМИ ОГРАНИЧЕНИЯМИ)
# =========================================================================
s = sp.Symbol('s', positive=True)
v0 = sp.Symbol('v0', positive=True)
v = sp.Symbol('v', positive=True)
a = sp.Symbol('a')
t = sp.Symbol('t', positive=True)

F, m, g, N, mu, F_fr, k, x = sp.symbols('F m g N mu F_fr k x')
m1 = sp.Symbol('m1', positive=True)
m2 = sp.Symbol('m2', positive=True)
T = sp.Symbol('T')
h, p, Ek, Ep = sp.symbols('h p Ek Ep')
v1, v2, u = sp.symbols('v1 v2 u', positive=True)
A, P_pow = sp.symbols('A P_pow')
alpha, tau = sp.symbols('alpha tau')

M_torque, I_inert, eps, omega, R_rad, L_am = sp.symbols('M_torque I_inert eps omega R_rad L_am')

# Термодинамика и МКТ
P_gas = sp.Symbol('P_gas', positive=True)
V_gas = sp.Symbol('V_gas', positive=True)
T_gas = sp.Symbol('T_gas', positive=True)
nu_moles = sp.Symbol('nu_moles', positive=True)
R_gas = sp.Symbol('R_gas', positive=True)
M_molar = sp.Symbol('M_molar', positive=True)

U_energy, Q_heat, A_gas, i_deg, eta_kpd = sp.symbols('U_energy Q_heat A_gas i_deg eta_kpd')

# Символы фазовых макросостояний (строго положительные)
T1 = sp.Symbol('T1', positive=True)
T2 = sp.Symbol('T2', positive=True)
V1 = sp.Symbol('V1', positive=True)
V2 = sp.Symbol('V2', positive=True)
P1 = sp.Symbol('P1', positive=True)
P2 = sp.Symbol('P2', positive=True)

k_spring = sp.Symbol('k_spring', positive=True)
x_stretch = sp.Symbol('x_stretch')
E_el = sp.symbols('E_el')
v_rms = sp.Symbol('v_rms', positive=True)
gamma_adiab = sp.Symbol('gamma_adiab', positive=True)
k_boltz = sp.Symbol('k_boltz', positive=True)
e_molek = sp.Symbol('e_molek', positive=True)

t_ind = sp.Symbol('t', real=True, positive=True)
x_ind = sp.Symbol('x', real=True)
x_t = sp.Function('x')(t_ind)
v_t = sp.Function('v')(t_ind)
a_t = sp.Function('a')(t_ind)

# Электростатика
q1_ch = sp.Symbol('q1_ch')
q2_ch = sp.Symbol('q2_ch')
q_charge = sp.Symbol('q_charge')
k_coulomb = sp.Symbol('k_coulomb', positive=True)
r_dist = sp.Symbol('r_dist', positive=True)
E_field = sp.Symbol('E_field')
U_volt = sp.Symbol('U_volt')
C_cap = sp.Symbol('C_cap', positive=True)
W_cap = sp.Symbol('W_cap', positive=True)

# Постоянный ток / электрические цепи
I_current = sp.Symbol('I_current')
R_res = sp.Symbol('R_res', positive=True)
R1_res = sp.Symbol('R1_res', positive=True)
R2_res = sp.Symbol('R2_res', positive=True)
P_el = sp.Symbol('P_el', positive=True)
Q_joule = sp.Symbol('Q_joule', positive=True)

# Магнетизм и электромагнитная индукция
B_field = sp.Symbol('B_field', positive=True)
F_lor = sp.Symbol('F_lor', positive=True)
F_amp = sp.Symbol('F_amp', positive=True)
l_wire = sp.Symbol('l_wire', positive=True)
S_area = sp.Symbol('S_area', positive=True)
Phi_flux = sp.Symbol('Phi_flux')
EMF_ind = sp.Symbol('EMF_ind')

# Колебания
T_period = sp.Symbol('T_period', positive=True)
freq = sp.Symbol('freq', positive=True)
A_ampl = sp.Symbol('A_ampl', positive=True)
l_pend = sp.Symbol('l_pend', positive=True)
W_osc = sp.Symbol('W_osc', positive=True)

# Волны
v_wave = sp.Symbol('v_wave', positive=True)
lambda_wave = sp.Symbol('lambda_wave', positive=True)

# Оптика
f_lens = sp.Symbol('f_lens')
d_obj = sp.Symbol('d_obj', positive=True)
d_img = sp.Symbol('d_img')
gamma_magnif = sp.Symbol('gamma_magnif')
n1_refr = sp.Symbol('n1_refr', positive=True)
n2_refr = sp.Symbol('n2_refr', positive=True)
angle_i = sp.Symbol('angle_i', positive=True)
angle_r = sp.Symbol('angle_r', positive=True)

# =========================================================================
# 2. СИСТЕМНЫЕ КАРТЫ И МАППИНГИ
# =========================================================================
SYMBOLS_MAP = {
    "s": s, "v0": v0, "v": v, "a": a, "t": t,
    "f": F, "m": m, "g": g, "n": N, "mu": mu, "f_fr": F_fr, "k": k, "x": x,
    "m1": m1, "m2": m2, "t_tension": T, "h": h, "p": p, "ek": Ek, "ep": Ep,
    "v1": v1, "v2": v2, "u": u, "a_work": A, "p_pow": P_pow, "alpha": alpha, "tau": tau,
    "m_torque": M_torque, "i_inert": I_inert, "eps": eps, "omega": omega, "r_rad": R_rad, "l_am": L_am,
    "p_gas": P_gas, "v_gas": V_gas, "t_gas": T_gas, "nu_moles": nu_moles, "r_gas": R_gas, "m_molar": M_molar,
    "u_energy": U_energy, "q_heat": Q_heat, "a_gas": A_gas, "i_deg": i_deg, "eta_kpd": eta_kpd,
    "t1": T1, "t2": T2, "v1_vol": V1, "v2_vol": V2, "p1_pres": P1, "p2_pres": P2,
    "k_spring": k_spring, "x_stretch": x_stretch, "e_el": E_el,
    "v_rms": v_rms, "gamma_adiab": gamma_adiab, "e_molek": e_molek,

    # Электростатика
    "q1_ch": q1_ch, "q2_ch": q2_ch, "q_charge": q_charge, "k_coulomb": k_coulomb,
    "r_dist": r_dist, "e_field": E_field, "u_volt": U_volt, "c_cap": C_cap, "w_cap": W_cap,

    # Постоянный ток
    "i_current": I_current, "r_res": R_res, "r1_res": R1_res, "r2_res": R2_res,
    "p_el": P_el, "q_joule": Q_joule,

    # Магнетизм и ЭМИ
    "b_field": B_field, "f_lor": F_lor, "f_amp": F_amp, "l_wire": l_wire,
    "s_area": S_area, "phi_flux": Phi_flux, "emf_ind": EMF_ind,

    # Колебания
    "t_period": T_period, "freq": freq, "a_ampl": A_ampl, "l_pend": l_pend, "w_osc": W_osc,

    # Волны
    "v_wave": v_wave, "lambda_wave": lambda_wave,

    # Оптика
    "f_lens": f_lens, "d_obj": d_obj, "d_img": d_img, "gamma_magnif": gamma_magnif,
    "n1_refr": n1_refr, "n2_refr": n2_refr, "angle_i": angle_i, "angle_r": angle_r
}

WORDS_MAP = {
    "ускорение": "a", "сила": "f", "время": "t", "путь": "s", "натяжение": "t_tension",
    "m1": "m1", "m2": "m2", "скорость": "v", "высота": "h", "импульс": "p", "энергия": "ek",
    "v1": "v1", "v2": "v2", "работа": "a_work", "мощность": "p_pow", "угол": "alpha",
    "трения": "f_fr", "коэффициент": "mu", "коэффициенте": "mu", "тау": "tau", "блок": "t_tension",
    "момент": "m_torque", "инерции": "i_inert", "радиус": "r_rad", "угловое": "eps",
    "давление": "p_gas", "объем": "v_gas", "температура": "t_gas", "моль": "nu_moles", "количество": "nu_moles",
    "масса": "m", "молярная": "m_molar", "теплота": "q_heat", "кпд": "eta_kpd",
    "нагревателя": "t1", "холодильника": "t2",
    "жесткость": "k_spring", "жесткостью": "k_spring", "удлинение": "x_stretch", "деформация": "x_stretch",
    "среднеквадратичная": "v_rms", "адиабаты": "gamma_adiab", "молекулы": "e_molek",

    # Электростатика
    "заряд": "q_charge", "заряды": "q_charge", "заряда": "q_charge",
    "расстояние": "r_dist", "напряженность": "e_field", "напряжение": "u_volt",
    "потенциал": "u_volt", "емкость": "c_cap", "конденсатор": "c_cap", "конденсатора": "c_cap",

    # Постоянный ток
    "ток": "i_current", "сопротивление": "r_res", "сопротивлением": "r_res",

    # Магнетизм
    "индукция": "b_field", "индукции": "b_field", "магнитного": "b_field",
    "виток": "s_area", "площадь": "s_area", "поток": "phi_flux", "эдс": "emf_ind",
    "провод": "l_wire", "проводника": "l_wire",

    # Колебания
    "период": "t_period", "частота": "freq", "амплитуда": "a_ampl",
    "маятника": "l_pend", "маятник": "l_pend", "колебаний": "t_period",

    # Волны
    "длина волны": "lambda_wave", "скорость волны": "v_wave",

    # Оптика
    "линза": "f_lens", "линзы": "f_lens", "фокусное": "f_lens",
    "изображение": "d_img", "предмет": "d_obj", "преломления": "n1_refr", "увеличение": "gamma_magnif"
}

for k in SYMBOLS_MAP.keys():
    WORDS_MAP[k] = k

UNITS_MAP = {
    "кг": "m", "ньютон": "f", "н": "f", "метров": "s", "с": "t", "секунд": "t",
    "мс2": "a", "м/с2": "a", "м/с": "v", "дж": "ek", "вт": "p_pow",
    "град": "alpha", "градусов": "alpha", "градусам": "alpha",
    "рад/с2": "eps", "рад/с": "omega",
    "па": "p_gas", "паскалей": "p_gas", "кельвин": "t_gas", "к": "t_gas",
    "моль": "nu_moles", "литров": "v_gas", "м3": "v_gas",
    "н/м": "k_spring", "мм": "x_stretch", "миллиметров": "x_stretch",

    # Электростатика / ток
    "кл": "q_charge", "кулон": "q_charge", "мкл": "q_charge", "нкл": "q_charge",
    "в": "u_volt", "вольт": "u_volt", "вольта": "u_volt",
    "ф": "c_cap", "мкф": "c_cap", "ом": "r_res", "а": "i_current", "ампер": "i_current",
    "н/кл": "e_field", "в/м": "e_field",

    # Магнетизм
    "тл": "b_field", "тесла": "b_field",

    # Колебания / волны
    "гц": "freq", "герц": "freq", "см": "a_ampl",

    # Оптика
    "дптр": "f_lens", "градус": "angle_i"
}

CONSTANTS = {g: 9.8, R_gas: 8.31, k_boltz: 1.38e-23, k_coulomb: 8.99e9}

NAMES_MAP = {
    "s": "Пройденный путь", "v0": "Начальная скорость", "v": "Текущая скорость",
    "a": "Ускорение", "t": "Время движения", "f": "Внешняя сила", "m": "Масса тела",
    "g": "Ускорение свободного падения", "n": "Сила реакции опоры", "mu": "Коэффициент трения",
    "f_fr": "Сила трения", "m1": "Масса 1", "m2": "Масса 2",
    "p_gas": "Давление газа", "v_gas": "Объем газа", "t_gas": "Температура газа",
    "nu_moles": "Количество вещества (моли)", "r_gas": "Универсальная газовая постоянная",
    "m_molar": "Молярная масса", "u_energy": "Внутренняя энергия газа", "q_heat": "Количество теплоты",
    "a_gas": "Работа газа", "i_deg": "Степени свободы молекул", "eta_kpd": "Коэффициент полезного действия (КПД)",
    "t1": "Температура состояния 1", "t2": "Температура состояния 2",
    "v1_vol": "Начальный объем (V₁)", "v2_vol": "Конечный объем (V₂)", "p1_pres": "Начальное давление (P₁)", "p2_pres": "Конечное давление (P₂)",
    "k_spring": "Жесткость пружины", "x_stretch": "Деформация пружины", "e_el": "Энергия упругой деформации",
    "v_rms": "Среднеквадратичная скорость молекул", "gamma_adiab": "Показатель адиабаты Пуассона", "e_molek": "Кинетическая энергия молекулы",

    # Электростатика
    "q1_ch": "Заряд 1", "q2_ch": "Заряд 2", "q_charge": "Заряд", "k_coulomb": "Постоянная Кулона",
    "r_dist": "Расстояние между зарядами", "e_field": "Напряженность электрического поля",
    "u_volt": "Напряжение (разность потенциалов)", "c_cap": "Электроёмкость", "w_cap": "Энергия конденсатора",

    # Постоянный ток
    "i_current": "Сила тока", "r_res": "Электрическое сопротивление",
    "r1_res": "Сопротивление 1", "r2_res": "Сопротивление 2",
    "p_el": "Электрическая мощность", "q_joule": "Количество теплоты (закон Джоуля-Ленца)",

    # Магнетизм
    "b_field": "Индукция магнитного поля", "f_lor": "Сила Лоренца", "f_amp": "Сила Ампера",
    "l_wire": "Длина проводника", "s_area": "Площадь контура", "phi_flux": "Магнитный поток",
    "emf_ind": "ЭДС индукции",

    # Колебания
    "t_period": "Период колебаний", "freq": "Частота колебаний", "a_ampl": "Амплитуда колебаний",
    "l_pend": "Длина маятника", "w_osc": "Полная энергия колебаний",

    # Волны
    "v_wave": "Скорость волны", "lambda_wave": "Длина волны",

    # Оптика
    "f_lens": "Фокусное расстояние линзы", "d_obj": "Расстояние до предмета",
    "d_img": "Расстояние до изображения", "gamma_magnif": "Увеличение линзы",
    "n1_refr": "Показатель преломления среды 1", "n2_refr": "Показатель преломления среды 2",
    "angle_i": "Угол падения", "angle_r": "Угол преломления"
}

EQUATIONS = {
    "Кинематика": [v0 * t + (a * t**2) / 2 - s, v0 + a * t - v],
    "Кинематика (Высшая математика)": [sp.Symbol("calculus_flag")],
    "Динамика (Одно тело)": [F - m * a, mu * N - F_fr, m * g - N, k_spring * x_stretch - F],
    "Динамика (Система тел)": [
        # Машина Атвуда: два груза на нити через блок, движение только за
        # счёт силы тяжести (m1 тяжелее и опускается, m2 поднимается).
        m1 * g - T - m1 * a,
        T - m2 * g - m2 * a,
        # Связанные тела под внешней силой (горизонтально, трением и
        # гравитацией в горизонтальном движении пренебрегаем).
        F - (m1 + m2) * a,
        T - m2 * a,
    ],
    "Законы сохранения": [m * v - p, (m * v**2) / 2 - Ek, F * s - A, m1 * v1 + m2 * v2 - (m1 + m2) * u, (k_spring * x_stretch**2) / 2 - E_el, m * g * h - Ek],
    "Динамика (Наклонная плоскость)": [m * g * sp.sin(alpha) - F_fr - m * a, N - m * g * sp.cos(alpha), mu * N - F_fr],
    "Динамика вращательного движения": [M_torque - I_inert * eps, F * R_rad - M_torque, omega / t - eps, v / R_rad - omega],
    "Молекулярная физика и Термодинамика": [
        P_gas * V_gas - nu_moles * R_gas * T_gas,
        P_gas * V_gas - (m / M_molar) * R_gas * T_gas,
        P1 * V1 - nu_moles * R_gas * T1,
        P2 * V2 - nu_moles * R_gas * T2,
        Q_heat - U_energy - A_gas,
        U_energy - (i_deg / 2) * nu_moles * R_gas * T_gas,
        A_gas - P_gas * (V2 - V1),
        eta_kpd - (T1 - T2) / T1,
        eta_kpd - A_gas / Q_heat,
        v_rms**2 * M_molar - 3 * R_gas * T_gas,
        e_molek - (3 / 2) * k_boltz * T_gas,
        P1 * V1**gamma_adiab - P2 * V2**gamma_adiab
    ],
    "Электростатика": [
        k_coulomb * q1_ch * q2_ch / r_dist**2 - F,
        k_coulomb * q_charge / r_dist**2 - E_field,
        q_charge * E_field - F,
        k_coulomb * q1_ch * q2_ch / r_dist - Ep,
        E_field * r_dist - U_volt,
        C_cap * U_volt - q_charge,
        (C_cap * U_volt**2) / 2 - W_cap,
    ],
    "Постоянный ток": [
        U_volt - I_current * R_res,
        P_el - I_current * U_volt,
        P_el - I_current**2 * R_res,
        R_res - (R1_res + R2_res),
        1 / R_res - (1 / R1_res + 1 / R2_res),
        Q_joule - I_current**2 * R_res * t,
    ],
    "Магнетизм и электромагнитная индукция": [
        F_lor - q_charge * v * B_field,
        F_amp - B_field * I_current * l_wire,
        Phi_flux - B_field * S_area,
        EMF_ind - Phi_flux / t,
        r_dist - (m * v) / (q_charge * B_field),
    ],
    "Колебания": [
        T_period - 2 * sp.pi * sp.sqrt(m / k_spring),
        T_period - 2 * sp.pi * sp.sqrt(l_pend / g),
        freq * T_period - 1,
        omega - 2 * sp.pi * freq,
        W_osc - (k_spring * A_ampl**2) / 2,
    ],
    "Волны": [
        v_wave - lambda_wave * freq,
    ],
    "Оптика": [
        1 / f_lens - (1 / d_obj + 1 / d_img),
        gamma_magnif - d_img / d_obj,
        n1_refr * sp.sin(angle_i) - n2_refr * sp.sin(angle_r),
    ],
}