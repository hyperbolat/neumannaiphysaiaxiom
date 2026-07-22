# -*- coding: utf-8 -*-
"""
Формирует пошаговый отчёт о решении задачи.

Что изменилось относительно предыдущей версии (по результатам прямой
жалобы пользователя — "объясняет реально плохо"):

  1. Комментарий к каждому уравнению раньше выбирался цепочкой if/elif на
     ~11 конкретных случаев, написанных под первые 8 разделов физики. Для
     всех 6 новых разделов (электростатика, ток, магнетизм, колебания,
     волны, оптика) комментарий ВСЕГДА был обезличенной заглушкой
     "Уравнение физического баланса...". Теперь используется таблица
     из equation_commentary.py, покрывающая ВСЕ ~50 уравнений явно и
     проверенно (см. коммент в том файле).

  2. Раньше отчёт показывал только абстрактный вид уравнения (например
     "M_torque = I_inert·eps") и сразу прыгал к финальному ответу — ни
     разу не показывая, откуда он взялся. Это не объяснение, а цитирование
     закона. Теперь для каждого шага показывается: закон -> уравнение с
     подставленными известными числами -> (если решается) промежуточный
     численный результат, который становится "известным" для следующего
     шага — так, как школьник/студент реально решал бы задачу вручную.

  3. Убрана маркетинговая лексика ("Нейросимволический Нейросимволический
     Аналитический Отчет", "Квантовый рендеринг", emoji-заголовки) —
     мешает читать, не помогает понять физику.

  4. _pretty_format переписан на таблицу символ->глиф вместо ручной
     цепочки .replace() на десяток переменных — раньше новые переменные
     (все из 6 новых разделов) просто не форматировались вообще.
"""

import re
import sympy as sp

from physics_db import NAMES_MAP, SYMBOLS_MAP, CONSTANTS
import equation_commentary as ec

# =========================================================================
# СИМВОЛ -> ГЛИФ ДЛЯ КРАСИВОГО ОТОБРАЖЕНИЯ УРАВНЕНИЙ
# =========================================================================
SYMBOL_GLYPHS = {
    "v0": "v<sub>0</sub>", "F_fr": "F<sub>тр</sub>", "m1": "m<sub>1</sub>", "m2": "m<sub>2</sub>",
    "Ek": "E<sub>к</sub>", "Ep": "E<sub>п</sub>", "v1": "v<sub>1</sub>", "v2": "v<sub>2</sub>",
    "P_pow": "P", "alpha": "α", "tau": "τ",
    "M_torque": "M", "I_inert": "I", "eps": "ε", "omega": "ω", "R_rad": "R", "L_am": "L",
    "P_gas": "P", "V_gas": "V", "T_gas": "T", "nu_moles": "ν", "R_gas": "R", "M_molar": "M",
    "U_energy": "U", "Q_heat": "Q", "A_gas": "A<sub>г</sub>", "i_deg": "i", "eta_kpd": "η",
    "T1": "T<sub>1</sub>", "T2": "T<sub>2</sub>", "V1": "V<sub>1</sub>", "V2": "V<sub>2</sub>",
    "P1": "P<sub>1</sub>", "P2": "P<sub>2</sub>",
    "k_spring": "k", "x_stretch": "x", "E_el": "E<sub>упр</sub>",
    "v_rms": "v<sub>кв</sub>", "gamma_adiab": "γ", "k_boltz": "k", "e_molek": "E<sub>0</sub>",
    "q1_ch": "q<sub>1</sub>", "q2_ch": "q<sub>2</sub>", "q_charge": "q", "k_coulomb": "k",
    "r_dist": "r", "E_field": "E", "U_volt": "U", "C_cap": "C", "W_cap": "W<sub>C</sub>",
    "I_current": "I", "R_res": "R", "R1_res": "R<sub>1</sub>", "R2_res": "R<sub>2</sub>",
    "P_el": "P", "Q_joule": "Q",
    "B_field": "B", "F_lor": "F<sub>Л</sub>", "F_amp": "F<sub>А</sub>", "l_wire": "l",
    "S_area": "S", "Phi_flux": "Φ", "EMF_ind": "ε<sub>i</sub>",
    "T_period": "T", "freq": "f", "A_ampl": "A", "l_pend": "l", "W_osc": "W",
    "v_wave": "v", "lambda_wave": "λ",
    "f_lens": "F", "d_obj": "d", "d_img": "f&#8242;", "gamma_magnif": "Γ",
    "n1_refr": "n<sub>1</sub>", "n2_refr": "n<sub>2</sub>", "angle_i": "α", "angle_r": "β",
}
# Сортировка по убыванию длины имени — чтобы "F_fr" подставлялось раньше "F"
# и не оставляло висящий "_fr" (хотя \b-границы и так это предотвращают,
# так надёжнее при любых будущих добавлениях).
_GLYPH_ORDER = sorted(SYMBOL_GLYPHS, key=len, reverse=True)


class PhysicsExplainer:
    def __init__(self):
        print("[Explainer] Готов.")

    # ------------------------------------------------------------------
    def _format_value(self, val):
        if isinstance(val, (int, float)):
            if val == 0:
                return "0.0"
            if abs(val) < 1e-3 or abs(val) > 1e6:
                return f"{val:.4e}"
            return f"{val:.4f}".rstrip('0').rstrip('.')
        return str(val)

    def _glyph_for_key(self, key: str) -> str:
        """Для отображения ключей словаря 'дано' (всегда нижний регистр,
        например 'i_inert', 'q1_ch') — а НЕ для строк уравнений sympy
        (там регистр уже правильный, 'I_inert'). Нельзя просто сделать
        поиск по глифам регистронезависимым: V1 (объём) и v1 (скорость
        первого тела) — разные физические величины, которые различаются
        только регистром, слияние по regex IGNORECASE исказило бы данные."""
        sym = SYMBOLS_MAP.get(key)
        if sym is None:
            base = re.sub(r'\d+$', '', key)
            idx = re.search(r'\d+$', key)
            if base in SYMBOLS_MAP and idx:
                sym = sp.Symbol(f"{SYMBOLS_MAP[base].name}{idx.group()}")
        name = sym.name if sym is not None else key
        return self._pretty_format(name)

    def _pretty_format(self, expr_str: str) -> str:
        # ВАЖНО: сначала подставляем имена символов (пока строка ещё в
        # чистом ASCII вида "I_current**2"), и только ПОТОМ делаем
        # косметику (**2 -> ², * -> ·). Если сделать наоборот, "²" сам
        # оказывается символом \w в юникодном режиме Python — граница \b
        # после "I_current²" не формируется, и замена по \bI_current\b
        # перестаёт срабатывать (найдено на живом тесте).
        res = expr_str
        for name in _GLYPH_ORDER:
            res = re.sub(rf'\b{re.escape(name)}\b', SYMBOL_GLYPHS[name], res)
        res = res.replace("**2", "²").replace("**3", "³").replace("*", "·")
        return res

    # ------------------------------------------------------------------
    def _build_known_values(self, data: dict) -> dict:
        """Строит {sympy.Symbol: число} из данных задачи + физических
        констант — используется, чтобы показать РЕАЛЬНУЮ подстановку
        чисел в уравнение, а не только его абстрактный вид."""
        known = {}
        for key, val in data.items():
            if not isinstance(val, (int, float)):
                continue
            if key in SYMBOLS_MAP:
                known[SYMBOLS_MAP[key]] = val
            else:
                base = re.sub(r'\d+$', '', key)
                idx = re.search(r'\d+$', key)
                if base in SYMBOLS_MAP and idx:
                    known[sp.Symbol(f"{SYMBOLS_MAP[base].name}{idx.group()}")] = val
        for c_sym, c_val in CONSTANTS.items():
            known[c_sym] = c_val
        return known

    def _round_expr_for_display(self, expr, sig: int = 4):
        """Округляет числовые литералы внутри sympy-выражения до sig
        значащих цифр — только для отображения (не для реального решения,
        там нужна полная точность). sp.Float хранит точность отдельно от
        значения: обычный Python round() тут не помогает (getattr
        sp.Float(15.0) печатается как '15.0000000000000' независимо от
        того, что "лишних" цифр у значения нет) — нужно явно задавать
        точность через sp.Float(val, sig)."""
        def _round_float(f):
            val = float(f)
            if val == 0:
                return sp.Float(0)
            return sp.Float(val, sig)
        replacements = {f: _round_float(f) for f in expr.atoms(sp.Float)}
        return expr.xreplace(replacements)

    @staticmethod
    def _trim_zeros(text: str) -> str:
        """'15.00' -> '15', '29.40' -> '29.4' — sp.Float(val, sig) всё
        равно печатает выровненное число знаков (дополняя нулями), это
        чистит финальную строку для отображения."""
        def _trim(m):
            s = m.group(0)
            return s.rstrip('0').rstrip('.') if '.' in s else s
        return re.sub(r'\d+\.\d+', _trim, text)

    def _lookup_commentary(self, eq):
        """Ищет (название закона, объяснение) в equation_commentary.py:
        сначала точное совпадение, потом — по базовому набору символов без
        индексов (для уравнений, клонированных под многотельные задачи)."""
        if eq in ec.EXACT_INDEX:
            return ec.EXACT_INDEX[eq]
        base = ec._base_symbol_set(eq)
        if base in ec.BASE_INDEX:
            return ec.BASE_INDEX[base]
        # Не должно происходить (таблица покрывает все уравнения из
        # physics_db), но на случай появления нового уравнения без записи
        # в таблице — не падаем, а честно описываем через известные имена.
        parts = [NAMES_MAP[s.name.lower()] for s in eq.free_symbols if s.name.lower() in NAMES_MAP]
        return "Уравнение", f"связывает параметры: {', '.join(parts)}."

    # ------------------------------------------------------------------
    def _progressive_steps(self, equations, data):
        """
        Проходит по уравнениям в том порядке, в котором их выбрал solver,
        и на каждом шаге пытается вычислить промежуточный числовой
        результат (тот единственный новый неизвестный, который вводит это
        уравнение) — в точности так, как решал бы человек: узнал одно,
        подставил дальше. known растёт с каждым шагом.

        Возвращает список dict: eq, law_name, law_text, substituted_str,
        solved_symbol_name (или None), solved_value (или None).
        """
        known = self._build_known_values(data)
        steps = []
        for eq in equations:
            law_name, law_text = self._lookup_commentary(eq)
            eq_substituted = eq.subs(known)
            free = eq_substituted.free_symbols

            solved_name, solved_value = None, None
            if len(free) == 1:
                target_sym = next(iter(free))
                try:
                    sols = sp.solve(eq_substituted, target_sym)
                    real_sols = [s for s in sols if getattr(s, "is_real", False)]
                    if real_sols:
                        val = float(real_sols[0])
                        known[target_sym] = val
                        solved_name, solved_value = target_sym.name, val
                except Exception:
                    pass

            if isinstance(eq, sp.Eq):
                eq_str = f"{eq.lhs} = {eq.rhs}"
            else:
                args = eq.args
                eq_str = f"{args[0]} = {-args[1]}" if len(args) == 2 else f"{eq} = 0"

            eq_disp = self._round_expr_for_display(eq_substituted)
            if solved_name is not None:
                # Один остающийся символ — показываем напрямую "символ = число",
                # а не сырое разложение sympy-аргументов (которое иногда даёт
                # режущее глаз "-3.84e-13 = -F" вместо "F = 3.84e-13").
                sub_str = f"{solved_name} = {self._format_value(solved_value)}"
            elif isinstance(eq_disp, sp.Eq):
                sub_str = self._trim_zeros(f"{eq_disp.lhs} = {eq_disp.rhs}")
            else:
                sargs = eq_disp.args
                sub_str = self._trim_zeros(f"{sargs[0]} = {-sargs[1]}" if len(sargs) == 2 else f"{eq_disp} = 0")

            is_redundant_check = (len(free) == 0)

            steps.append({
                "law_name": law_name,
                "law_text": law_text,
                "eq_str": eq_str,
                "sub_str": sub_str,
                "has_substitution": eq_substituted != eq,
                "is_redundant_check": is_redundant_check,
                "solved_name": solved_name,
                "solved_value": solved_value,
            })
        return steps

    # ------------------------------------------------------------------
    def generate_report(self, category: str, data: dict, target: str, result, actual_eqs=None):
        target_name = NAMES_MAP.get(target, target)
        used_equations = actual_eqs if actual_eqs else []
        total_stages = len(used_equations)
        steps = self._progressive_steps(used_equations, data)

        html = []
        html.append(
            "<div style='font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, sans-serif; "
            "color: #ffffff; line-height: 1.6;'>")
        html.append(
            "<h2 style='color: #3b82f6; border-bottom: 1px solid #334155; padding-bottom: 10px; "
            "text-align: left; font-size: 18px; font-weight: 600;'>Решение</h2>")

        html.append("<div style='margin-bottom: 20px;'>")
        html.append(
            "<span style='color: #94a3b8; font-size: 11px; font-weight: 700; letter-spacing: 1px;'>"
            "РАЗДЕЛ</span><br>")
        html.append(f"<span style='font-size: 14px; color: #34d399;'><b>{category}</b></span></div>")

        html.append("<div style='margin-bottom: 20px;'>")
        html.append(
            "<span style='color: #94a3b8; font-size: 11px; font-weight: 700; letter-spacing: 1px;'>"
            "ДАНО</span><br>")
        html.append("<table style='width: 100%; margin-top: 5px; font-size: 13px; border-collapse: collapse;'>")
        for key, val in data.items():
            html.append(
                f"<tr style='border-bottom: 1px solid #1e293b;'>"
                f"<td style='padding: 4px 0; color: #cbd5e1;'>{NAMES_MAP.get(key, key)}</td>"
                f"<td style='text-align: right; color: #34d399; font-weight: 600;'>"
                f"{self._glyph_for_key(key)} = {self._pretty_format(str(val))}</td></tr>")

        pretty_target = self._glyph_for_key(target)
        html.append(
            f"<tr><td style='padding: 6px 0; color: #ff9800; font-weight: bold;'>Найти</td>"
            f"<td style='text-align: right; color: #ff9800; font-weight: bold;'>"
            f"{target_name} ({pretty_target})</td></tr>")
        html.append("</table></div>")

        if steps:
            html.append("<div style='margin-bottom: 20px;'>")
            html.append(
                "<span style='color: #94a3b8; font-size: 11px; font-weight: 700; letter-spacing: 1px;'>"
                "ХОД РЕШЕНИЯ</span><br>")

            for i, st in enumerate(steps, 1):
                html.append(
                    "<div style='background-color: #1e293b; padding: 14px; margin-top: 8px; "
                    "border-radius: 6px; border-left: 4px solid #2563eb;'>")

                html.append(
                    f"<div style='color: #94a3b8; font-size: 11px; margin-bottom: 6px;'>"
                    f"Шаг {i} из {total_stages} &middot; <b style='color:#cbd5e1;'>{st['law_name']}</b></div>")

                html.append(
                    f"<div style='color: #38bdf8; font-family: \"Courier New\", monospace; font-size: 15px; "
                    f"font-weight: bold;'>{self._pretty_format(st['eq_str'])}</div>")

                html.append(
                    f"<div style='color: #cbd5e1; font-size: 13px; margin-top: 6px;'>{st['law_text']}</div>")

                if st["is_redundant_check"]:
                    html.append(
                        "<div style='color: #64748b; font-size: 12px; margin-top: 8px; font-style: italic;'>"
                        "Эта связь уже выполняется автоматически при уже найденных значениях "
                        "(дополнительная проверка согласованности).</div>")
                elif st["has_substitution"] and st["solved_name"] is None:
                    html.append(
                        f"<div style='color: #64748b; font-family: \"Courier New\", monospace; font-size: 13px; "
                        f"margin-top: 8px;'>Подставляем известное: "
                        f"<span style='color:#e2e8f0;'>{self._pretty_format(st['sub_str'])}</span></div>")

                if st["solved_name"] is not None:
                    base_name = re.sub(r'\d+$', '', st["solved_name"]).lower()
                    display_name = NAMES_MAP.get(base_name, st["solved_name"])
                    html.append(
                        f"<div style='color: #10b981; font-size: 13px; margin-top: 6px;'>"
                        f"&rarr; {display_name}: "
                        f"<b>{self._pretty_format(st['solved_name'])} = {self._format_value(st['solved_value'])}</b>"
                        f"</div>")

                html.append("</div>")
            html.append("</div>")

        html.append(
            "<div style='background-color: #1e293b; padding: 15px; border-left: 4px solid #10b981; "
            "border-radius: 6px; margin-top: 20px;'>")
        html.append(
            "<span style='color: #94a3b8; font-size: 11px; font-weight: 700; letter-spacing: 1px;'>"
            "ОТВЕТ</span><br>")

        result_str = str(result)
        if "Недостаточно" in result_str or "Не удалось" in result_str:
            html.append(
                f"<div style='color: #f43f5e; font-size: 14px; font-weight: bold; margin-top: 5px;'>"
                f"⚠ {result}</div>")
        elif "Выражено аналитически" in result_str:
            raw_formula = result_str.split("=")[-1].strip()
            html.append(
                "<div style='color: #cbd5e1; font-size: 13px; margin-top: 5px;'>"
                "Не все переменные удалось свести к числу — часть данных отсутствует. "
                "Получена формула через оставшиеся неизвестные:</div>")
            html.append(
                f"<div style='font-size: 18px; color: #10b981; font-weight: bold; margin-top: 5px;'>"
                f"{self._glyph_for_key(target)} = {self._pretty_format(raw_formula)}</div>")
        else:
            html.append(
                f"<div style='font-size: 22px; color: #10b981; font-weight: bold; margin-top: 5px;'>"
                f"{self._glyph_for_key(target)} = {self._format_value(result)}</div>")
        html.append("</div></div>")

        return "".join(html)