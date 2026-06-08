import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import os
import io
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ─────────────────────────────────────────────
st.set_page_config(
    page_title="배터리 Second-Life 추천 플랫폼",
    page_icon="🔋",
    layout="wide"
)

st.markdown("""
<style>
    .main-title   { font-size:28px; font-weight:700; margin-bottom:4px; }
    .sub-title    { font-size:14px; color:#888; margin-bottom:24px; }
    .metric-card  { background:#1a1a2e; border-radius:12px; padding:20px;
                    text-align:center; border:1px solid #2a2a4a; }
    .metric-val   { font-size:28px; font-weight:700; color:#00d4aa; }
    .metric-label { font-size:12px; color:#aaa; margin-top:4px; }
    .rec-card     { background:#1a1a2e; border-radius:12px; padding:16px 20px;
                    margin-bottom:10px; border:1px solid #2a2a4a; }
    .top-card     { border:2px solid #00d4aa !important; }
    .section-title{ font-size:18px; font-weight:600; margin:20px 0 12px; }
    .ref-text     { font-size:11px; color:#666; margin-top:4px; }
    .mode-badge   { display:inline-block; padding:3px 10px; border-radius:20px;
                    font-size:12px; font-weight:600; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# 배터리 특성 상수
#
# [SOH 3단계 판정 기준] PMC11033388
#   SOH > 80%     → 재사용 (Reuse)
#   50% < SOH ≤ 80% → 재활용 (Repurpose)
#   SOH ≤ 50%    → 해체 (Recycle)
#
# [사이클 수명] Ali et al. (2023), Section 2 p.2
#   LFP  4,000회 이상 / NCM 2,000회 / NCA 1,500회 / LCO 800회
#
# [캘린더 열화율] Ali et al. (2023), Section 3
#   LFP  < 1%/년 / NCM·NCA ~2%/년 / LCO ~3%/년
#
# [LFP 2차 수명] chrismi.sdsu.edu/publications/225.pdf
# ═══════════════════════════════════════════════
BAT_PROPS = {
    "NCM": dict(cycle_life=2000, nominal_v=3.6,  calendar_aging_rate_pct=2.0),
    "LFP": dict(cycle_life=4000, nominal_v=3.2,  calendar_aging_rate_pct=1.0,
                eis_threshold=60,
                second_life_cycles=(5000, 10000),
                second_life_years=(14, 28)),
    "NCA": dict(cycle_life=1500, nominal_v=3.6,  calendar_aging_rate_pct=2.0),
    "LCO": dict(cycle_life=800,  nominal_v=3.7,  calendar_aging_rate_pct=3.0),
}

# 배터리별 내부저항 기준값 (NASA PCoE B0005~B0018 실측 + Ali et al. 2023)
BAT_RESISTANCE = {
    "NCM": dict(r0=0.10, r_max=0.22),
    "LFP": dict(r0=0.08, r_max=0.16),
    "NCA": dict(r0=0.12, r_max=0.28),
    "LCO": dict(r0=0.15, r_max=0.30),
}

BAT_ENC = {"NCM": 0, "LFP": 1, "NCA": 2, "LCO": 3}

# ─────────────────────────────────────────────
# 배터리별 페널티 가중치
# 배터리 화학 특성에 따른 열화 속도 차이 반영
# cycle_multiplier: 사이클비율 × N% 페널티
# age_multiplier:   연수 × N% 페널티
# ─────────────────────────────────────────────
PENALTY_WEIGHTS = {
    "LFP": {"cycle_multiplier": 15, "age_multiplier": 1.0, "desc": "안정적 화학 (LiFePO₄)"},
    "NCM": {"cycle_multiplier": 20, "age_multiplier": 2.0, "desc": "표준 니켈-코발트-망간"},
    "NCA": {"cycle_multiplier": 22, "age_multiplier": 2.5, "desc": "고에너지 니켈-코발트-알루미늄"},
    "LCO": {"cycle_multiplier": 25, "age_multiplier": 3.0, "desc": "고에너지 리튬-코발트 (불안정)"},
}

def get_soh_tier(soh):
    if soh > 80:   return "reuse"
    elif soh > 50: return "repurpose"
    else:          return "recycle"

TIER_META = {
    "reuse":     ("재사용 (Reuse)",     "#00d4aa"),
    "repurpose": ("재활용 (Repurpose)", "#f0a500"),
    "recycle":   ("해체 (Recycle)",     "#e05555"),
}

# ═══════════════════════════════════════════════
# 1. EIS 모델 (Warwick DIB Dataset 기반)
#
# 학습 데이터: Rashid et al. (2023), doi:10.1016/j.dib.2023.109157
#   - Warwick DIB EIS 데이터셋 360개 파일
#   - 파일명에서 SOH 레이블 추출 (예: "80SOH.xls" → 80%)
#   - 입력 피처 15개: EIS 임피던스 스펙트럼 기반 특성값
#
# 피처 설계:
#   [0] Re (전해질 저항, 고주파 실수부)
#   [1] 최대 Z' / [2] 최소 Z'' / [3] 최대 Z'' / [4] 평균 Z' / [5] Z' 표준편차
#   [6] Rct (반원 직경 = Z'_max - Z'_min): Niri et al. (2022) 핵심 지표
#   [7] 반원 높이 / [8] 반원 면적
#   [9] 최대|Z| / [10] 평균|Z| / [11] 표준편차|Z|
#   [12] D-value (반원 완전도) / [13] 최대|Z''| / [14] 평균|Z''|
#
# 모델: GradientBoosting + RandomForest 앙상블
#   최종 SOH = (GB예측 + RF예측) / 2
# ═══════════════════════════════════════════════
def parse_xls_eis(file_input):
    """
    Warwick DIB 포맷 .xls 파일에서 EIS 임피던스 데이터 추출
    컬럼 구조: A=주파수(Hz), B=Z'(실수부,Ω), C=Z''(허수부,Ω)
    고주파 → 저주파 순으로 정렬하여 반환
    """
    if isinstance(file_input, (str, os.PathLike)):
        with open(file_input, 'rb') as f:
            raw = f.read()
    else:
        raw = file_input
    try:
        import xlrd
        wb = xlrd.open_workbook(file_contents=raw) if isinstance(raw, bytes) \
             else xlrd.open_workbook(file_input)
        ws = wb.sheet_by_index(0)
        freq_list, z_real_list, z_imag_list = [], [], []
        for row in range(ws.nrows):
            try:
                if ws.ncols >= 3:
                    freq = float(ws.cell_value(row, 0))
                    zr   = float(ws.cell_value(row, 1))
                    zi   = float(ws.cell_value(row, 2))
                    if freq > 0:
                        freq_list.append(freq)
                        z_real_list.append(zr)
                        z_imag_list.append(zi)
            except:
                continue
        if z_real_list:
            idx = np.argsort(freq_list)[::-1]
            return [z_real_list[i] for i in idx], [z_imag_list[i] for i in idx]
    except ImportError:
        st.warning("⚠️ xlrd 설치 필요: pip install xlrd>=2.0.1")
    except Exception as e:
        print(f"XLS 파싱 오류: {e}")
    return [], []

def parse_csv_eis(file_input):
    """CSV 포맷 EIS 파일 파싱 (freq, z_real, z_imag 3열)"""
    df = pd.read_csv(io.BytesIO(file_input) if isinstance(file_input, bytes)
                     else file_input, header=None)
    df.columns = ['freq', 'z_real', 'z_imag']
    df = df.sort_values('freq', ascending=False).reset_index(drop=True)
    return df['z_real'].tolist(), df['z_imag'].tolist()

def extract_eis_features(z_real_list, z_imag_list):
    """
    EIS 데이터 → ML 피처 15개 추출
    핵심 지표인 Rct(반원 직경)는 배터리 열화와 가장 강한 상관을 가짐
    근거: Niri et al. (2022), Warwick DIB 데이터셋 피처 분석
    """
    if len(z_real_list) < 5:
        return None
    zr = np.array(z_real_list)
    zi = np.array(z_imag_list) if z_imag_list else np.zeros_like(zr)
    Rct         = zr.max() - zr[0]
    semi_height = abs(zi.min())
    semi_area   = np.pi * (Rct/2) * (semi_height/2) if Rct > 0 and semi_height > 0 else 0
    Z_mag       = np.sqrt(zr**2 + zi**2)
    D_value     = (Rct**2 - 4*semi_height**2) / (Rct**2 + 4*semi_height**2) \
                  if (Rct**2 + 4*semi_height**2) > 0 else 0
    return [
        float(zr[0]), float(zr.max()), float(zi.min()), float(zi.max()),
        float(zr.mean()), float(zr.std()),
        float(Rct), float(semi_height), float(semi_area),
        float(np.max(Z_mag)), float(np.mean(Z_mag)), float(np.std(Z_mag)),
        float(D_value), float(np.max(np.abs(zi))), float(np.mean(np.abs(zi))),
    ]

@st.cache_resource
def train_eis_model():
    """
    Warwick DIB EIS 데이터셋으로 SOH 예측 모델 학습
    Cell02_95SOH_15degC_05SOC_9505.zip 파일 로드 (루트 또는 data 폴더)
    """
    import zipfile
    base_dir = os.path.dirname(__file__)
    
    # 찾을 파일 경로 (우선순위 순)
    zip_candidates = [
        os.path.join(base_dir, 'Cell02_95SOH_15degC_05SOC_9505.zip'),
        os.path.join(base_dir, 'data', 'Cell02_95SOH_15degC_05SOC_9505.zip'),
        os.path.join(base_dir, 'data', 'EIS_Test.zip'),
        os.path.join(base_dir, 'EIS_Test.zip'),
        os.path.join(base_dir, 'data', 'EIS_Test'),
    ]

    file_items = []
    
    # zip 파일 탐색
    for zip_path in zip_candidates:
        if os.path.isfile(zip_path) and zip_path.endswith('.zip'):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for zname in zf.namelist():
                        fname = os.path.basename(zname)
                        if fname.endswith('.xls') and 'SOH' in fname:
                            file_items.append((fname, zf.read(zname)))
                if file_items:
                    break
            except Exception as e:
                continue
        elif os.path.isdir(zip_path):
            for fname in os.listdir(zip_path):
                if fname.endswith('.xls') and 'SOH' in fname:
                    file_items.append((fname, os.path.join(zip_path, fname)))
            if file_items:
                break
    
    if not file_items:
        return None, 0, 0, 0

    X, y = [], []
    for fname, file_data in file_items:
        m = re.search(r'(\d+)SOH', fname)
        if not m: continue
        soh = int(m.group(1))
        try:
            raw = file_data if isinstance(file_data, bytes) else None
            zr, zi = parse_xls_eis(raw if raw else file_data)
            feats = extract_eis_features(zr, zi)
            if feats:
                X.append(feats); y.append(soh)
        except:
            continue

    if len(X) < 10:
        return None, len(X), 0, 0

    X, y   = np.array(X), np.array(y)
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    gb = GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    gb.fit(Xs, y); rf.fit(Xs, y)

    cv_gb = cross_val_score(gb, Xs, y, cv=5, scoring='r2').mean()
    cv_rf = cross_val_score(rf, Xs, y, cv=5, scoring='r2').mean()

    return {'gb': gb, 'rf': rf, 'scaler': scaler}, len(X), cv_gb, cv_rf

def predict_soh_eis(models, zr, zi):
    """EIS ML 앙상블 예측 (GB + RF 평균)"""
    feats = extract_eis_features(zr, zi)
    if feats is None: return None
    Xs   = models['scaler'].transform([feats])
    pred = (models['gb'].predict(Xs)[0] + models['rf'].predict(Xs)[0]) / 2
    return round(float(np.clip(pred, 50, 100)), 1)

# ═══════════════════════════════════════════════
# 2. BMS 모델 (수동 입력 전용)
#
# 학습 데이터 설계 근거:
#   배터리 종류별 사이클 수명: Ali et al. (2023) Section 2
#   캘린더 열화율:             Ali et al. (2023) Section 3
#   내부저항 범위:             NASA PCoE B0005~B0018 실측
#
# SOH 열화 모델:
#   SOH = 100 - 20×min(cycle/cycle_life, 1.5) - cal_rate×years
#   근거: 설계 수명 도달 시 SOH 80% (80% rule, Ali et al. 2023)
#
# 피처 9개:
#   cycle_count, years, internal_resistance_ohm,
#   charge_time_s, temp_rise_c, voltage_drop_v,
#   coulombic_efficiency, bat_type_enc, cycle_life
#
# 핵심 3개 입력(사이클·연수·내부저항) → SOH 역산 → 나머지 피처 물리 모델 계산
# 이유: 나머지를 고정값으로 주면 ML이 사이클/연수를 무시하는 문제 발생
# ═══════════════════════════════════════════════
@st.cache_resource
def train_bms_model():
    """
    배터리 종류별 열화 특성 반영 BMS SOH 예측 모델
    NCM/LFP/NCA/LCO 각 500개 = 총 2,000개 합성 데이터
    사이클: 0 ~ cycle_life × 1.5 (수명 초과 케이스 포함)
    """
    np.random.seed(42)
    N_per = 500
    all_X, all_y = [], []

    for bat, cfg in BAT_PROPS.items():
        cl  = cfg['cycle_life']
        cr  = cfg.get('calendar_aging_rate_pct', 2.0)
        r0  = BAT_RESISTANCE[bat]['r0']
        rm  = BAT_RESISTANCE[bat]['r_max']

        cycle = np.random.randint(0, int(cl * 1.5), N_per).astype(float)
        years = np.random.uniform(0, 15, N_per)

        # SOH 열화 모델 (Ali et al. 2023)
        soh_cycle = 100 - 20 * np.clip(cycle / cl, 0, 1.5)
        soh_cal   = cr * years
        soh = np.clip(soh_cycle - soh_cal + np.random.normal(0, 1.5, N_per), 50, 100)

        # 내부 저항: 사이클비에 비례 r0→rm (NASA PCoE 실측 패턴)
        int_r  = r0 + (rm-r0)*np.clip(cycle/cl, 0, 1.3) + np.random.normal(0, 0.004, N_per)
        int_r  = np.clip(int_r, r0, rm*1.3)
        chg_t  = 3600*(1+(100-soh)/300) + np.random.normal(0, 100, N_per)
        t_rise = 5+(100-soh)/12 + np.random.normal(0, 0.5, N_per)
        v_drop = 0.05+(100-soh)/1200 + np.random.normal(0, 0.004, N_per)
        c_eff  = 0.99-(100-soh)/6000 + np.random.normal(0, 0.002, N_per)
        bat_f  = np.full(N_per, float(BAT_ENC[bat]))
        cl_f   = np.full(N_per, float(cl))

        X = np.column_stack([cycle, years, int_r, chg_t, t_rise, v_drop, c_eff, bat_f, cl_f])
        all_X.append(X); all_y.append(soh)

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X_all)

    gb = GradientBoostingRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.04,
        subsample=0.8, random_state=42
    )
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    gb.fit(Xs, y_all); rf.fit(Xs, y_all)

    cv_gb = cross_val_score(gb, Xs, y_all, cv=5, scoring='r2').mean()
    cv_rf = cross_val_score(rf, Xs, y_all, cv=5, scoring='r2').mean()

    return {'gb': gb, 'rf': rf, 'scaler': scaler}, cv_gb, cv_rf

def build_bms_features(bat_type, cycle, years, int_r):
    """
    핵심 3개 입력으로 전체 9개 피처 자동 계산
    사이클+연수 60% + 내부저항 40% 가중 평균으로 SOH 추정 후 물리 역산
    """
    cfg = BAT_PROPS[bat_type]
    cl, cr = cfg['cycle_life'], cfg.get('calendar_aging_rate_pct', 2.0)
    r0 = BAT_RESISTANCE[bat_type]['r0']
    rm = BAT_RESISTANCE[bat_type]['r_max']

    soh_cycle  = 100 - 20 * min(cycle / cl, 1.5)
    soh_theory = max(50.0, soh_cycle - cr * years)
    r_ratio    = (int_r - r0) / (rm - r0) if (rm - r0) > 0 else 0
    soh_from_r = max(50.0, 100 - 20 * min(r_ratio, 1.3))
    soh_est    = soh_theory * 0.6 + soh_from_r * 0.4

    chg_t  = 3600 * (1 + (100 - soh_est) / 300)
    t_rise = 5 + (100 - soh_est) / 12
    v_drop = 0.05 + (100 - soh_est) / 1200
    c_eff  = 0.99 - (100 - soh_est) / 6000

    return [float(cycle), float(years), float(int_r),
            chg_t, t_rise, v_drop, c_eff,
            float(BAT_ENC[bat_type]), float(cl)]

def predict_soh_bms(models, bat_type, cycle, years, int_r):
    """BMS 수동 입력 → SOH 예측 (앙상블)"""
    feats = build_bms_features(bat_type, cycle, years, int_r)
    Xs    = models['scaler'].transform([feats])
    pred  = (models['gb'].predict(Xs)[0] + models['rf'].predict(Xs)[0]) / 2
    return round(float(np.clip(pred, 50, 100)), 1)

# ═══════════════════════════════════════════════
# 3. NASA .mat 파일 파싱 (배치 처리 전용)
#
# 출처: NASA PCoE Battery Dataset
#   ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
# ═══════════════════════════════════════════════
def parse_nasa_mat(file_bytes):
    """
    NASA PCoE .mat 파일 파싱 (B0005~B0018 실제 구조 기반)
    확인된 구조:
      mat['B000X']       → shape (1,1), dtype [('cycle','O')]
      mat['B000X'][0,0]  → numpy.void, field: 'cycle'
      ['cycle']          → shape (1,N)
      ['cycle'][0,i]     → numpy.void
        ['type']         → shape (1,), dtype '<U9' → flat[0] = 'charge'/'discharge'/'impedance'
        ['data']         → shape (1,1) → [0,0] = numpy.void (측정값)
          ['Capacity']   → shape (1,1) float64 (discharge 전용, 마지막값 = 총 방전용량)
          ['Re']         → shape (1,1) float64 (impedance 전용)
    """
    try:
        import scipy.io
        mat      = scipy.io.loadmat(io.BytesIO(file_bytes), simplify_cells=False)
        bat_name = [k for k in mat.keys() if not k.startswith('_')][0]
        b0       = mat[bat_name][0, 0]
        cyc      = b0['cycle']            # shape (1, N)
        results  = []

        for i in range(cyc.shape[1]):
            try:
                c     = cyc[0, i]
                ctype = str(c['type'].flat[0]).strip().lower()
                data  = c['data'][0, 0]

                def get_arr(field, _d=data):
                    try:
                        return np.array(_d[field]).flatten().astype(float)
                    except Exception:
                        return np.array([])

                v  = get_arr('Voltage_measured')
                a  = get_arr('Current_measured')
                t  = get_arr('Temperature_measured')
                ts = get_arr('Time')

                entry = {
                    'cycle_idx':    i,
                    'type':         ctype,
                    'capacity_ah':  None,
                    'internal_r':   None,
                    'voltage_mean': float(np.mean(v))         if len(v)  > 0 else 0.0,
                    'current_mean': float(np.mean(np.abs(a))) if len(a)  > 0 else 0.0,
                    'temp_mean':    float(np.mean(t))         if len(t)  > 0 else 25.0,
                    'charge_time_s':float(ts[-1] - ts[0])     if len(ts) > 1 else 0.0,
                }
                if ctype == 'discharge':
                    cap = get_arr('Capacity')
                    if len(cap) > 0:
                        entry['capacity_ah'] = float(cap.flat[-1])
                if ctype == 'impedance':
                    re = get_arr('Re')
                    if len(re) > 0:
                        entry['internal_r'] = float(re.flat[0])

                results.append(entry)
            except Exception:
                continue
        return results, bat_name

    except Exception:
        pass

    # HDF5 fallback (v7.3 포맷)
    try:
        import h5py, tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            with h5py.File(tmp_path, 'r') as f:
                bat_name = [k for k in f.keys() if not k.startswith('#')][0]
                return _parse_hdf5_cycles(f[bat_name]), bat_name
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise ValueError(f"mat 파일 파싱 실패: {e}")



def _parse_hdf5_cycles(bat_group):
    import h5py
    results = []
    cycle_group = bat_group.get('cycle', bat_group)
    for i, key in enumerate(cycle_group.keys()):
        try:
            c = cycle_group[key]
            ctype_raw = c.get('type', None)
            if ctype_raw is None: continue
            if isinstance(ctype_raw, h5py.Dataset):
                ctype = ''.join(chr(x) for x in ctype_raw[()]).strip().lower()
            else:
                ctype = str(ctype_raw).strip().lower()
            data  = c.get('data', c)
            entry = {'cycle_idx': i, 'type': ctype, 'capacity_ah': None, 'internal_r': None}
            def get_arr(key):
                d = data.get(key, None)
                return np.array([]) if d is None else np.array(d[()])
            v  = get_arr('Voltage_measured')
            a  = get_arr('Current_measured')
            t  = get_arr('Temperature_measured')
            ts = get_arr('Time')
            entry['voltage_mean']  = float(np.mean(v))  if len(v)  else 0.0
            entry['current_mean']  = float(np.mean(np.abs(a))) if len(a) else 0.0
            entry['temp_mean']     = float(np.mean(t))  if len(t)  else 25.0
            entry['charge_time_s'] = float(ts[-1]-ts[0]) if len(ts) > 1 else 0.0
            if ctype == 'discharge':
                cap = data.get('Capacity', None)
                if cap is not None:
                    entry['capacity_ah'] = float(np.array(cap[()]).flat[0])
            if ctype == 'impedance':
                re = data.get('Re', None)
                if re is not None:
                    entry['internal_r'] = float(np.array(re[()]).flat[0])
            results.append(entry)
        except:
            continue
    return results

def compute_soh_from_mat(cycles_data):
    """
    방전 사이클 Capacity로 SOH 계산
    SOH = 현재 방전 용량 / 초기 방전 용량 × 100
    출처: NASA PCoE 데이터셋 정의
    
    ※ NASA 데이터 특성:
       - Capacity 값이 노이즈로 인해 불규칙함
       - 따라서 모든 방전 용량의 최대값을 "정격 용량"으로 설정
       - (이상값: 0, 매우 작은 값은 제외)
    """
    discharge = [c for c in cycles_data
                 if c['type'] == 'discharge' and c['capacity_ah'] is not None
                 and c['capacity_ah'] > 0.5]  # 0값, 이상값 제외 (0.5 Ah 이상)
    
    if not discharge: return []
    
    # 정격 용량 = 모든 방전 용량의 최대값
    # (초기 배터리 상태를 나타내므로 가장 높은 값이 맞음)
    rated_cap = max([c['capacity_ah'] for c in discharge])
    
    result = []
    for i, c in enumerate(discharge):
        soh = min(c['capacity_ah'] / rated_cap * 100, 100)  # 100% 초과 방지
        result.append({
            'discharge_idx': i, 'cycle_idx': c['cycle_idx'],
            'capacity_ah': round(c['capacity_ah'], 4), 'soh_actual': round(soh, 2),
            'voltage_mean': round(c['voltage_mean'], 3), 'current_mean': round(c['current_mean'], 3),
            'temp_mean': round(c['temp_mean'], 1), 'charge_time_s': round(c['charge_time_s'], 0),
        })
    
    impedance = [c for c in cycles_data
                 if c['type'] == 'impedance' and c['internal_r'] is not None]
    if impedance:
        imp_idx = np.array([c['cycle_idx'] for c in impedance])
        imp_r   = np.array([c['internal_r'] for c in impedance])
        for row in result:
            nearest = np.argmin(np.abs(imp_idx - row['cycle_idx']))
            row['internal_r'] = float(imp_r[nearest])
    else:
        for row in result:
            row['internal_r'] = None
    return result

# ═══════════════════════════════════════════════
# 공통 유틸: 활용처 추천 + 안전성 평가
# ═══════════════════════════════════════════════
def get_recommendations(health, years, cycles, bat_type, voltage):
    """
    배터리 2차 수명 활용처 추천 (배터리별 차등 페널티 적용)
    ─────────────────────────────────────────────
    조정된 SOH = 기본 SOH - 사이클 페널티 - 연수 페널티 - 전압 페널티

    페널티는 배터리 화학 특성에 따라 차등 적용:
    - LFP: 사이클 15%, 연 1%  (안정적)
    - NCM: 사이클 20%, 연 2%  (표준)
    - NCA: 사이클 22%, 연 2.5% (불안정)
    - LCO: 사이클 25%, 연 3%  (매우 불안정)

    활용처별 최소 SOH (조정값 기준):
    - 태양광 연계 ESS: 70% (Edge et al. 2023)
    - 가정용 ESS:      70% (Edge et al. 2023; UL 1974)
    - 통신기지국 백업: 60% (Martinez-Laserna et al. 2018)
    - UPS 비상전원:    50% (Edge et al. 2023)
    - 전기차 보조:     50% (Edge et al. 2023)
    ─────────────────────────────────────────────
    """
    props   = BAT_PROPS[bat_type]
    weights = PENALTY_WEIGHTS[bat_type]

    cycle_ratio   = cycles / props['cycle_life']
    cycle_penalty = min(cycle_ratio * weights['cycle_multiplier'], 25)
    age_penalty   = min(years * weights['age_multiplier'], 20)
    v_diff        = abs(voltage - props['nominal_v'])
    v_penalty     = min((v_diff / 0.3) * 10, 10) if v_diff > 0 else 0

    adjusted_health = health - cycle_penalty - age_penalty - v_penalty

    apps = [
        {
            "name": "전력망 연계 ESS (Grid ESS)", "icon": "🔋",
            "desc": "태양광/풍력 연계. 일일 1~2회 충방전. 5~10년 운영 기대.",
            "ref": "Edge et al. (2023); IEC 62933",
            "score": max(0, adjusted_health - 10),
            "condition": adjusted_health >= 70,
        },
        {
            "name": "태양광 주택용 ESS", "icon": "☀️",
            "desc": "가정용 태양광 연계 저장. 낮은 C-rate, 25년 설계수명.",
            "ref": "Edge et al. (2023); IEC 62933",
            "score": max(0, adjusted_health - 5),
            "condition": adjusted_health >= 70,
        },
        {
            "name": "무정전전원장치 (UPS)", "icon": "⚡",
            "desc": "비상/백업 전원. 간헐적 방전. 낮은 사이클 스트레스.",
            "ref": "Edge et al. (2023), PMC11033388",
            "score": max(0, adjusted_health),
            "condition": adjusted_health >= 50,
        },
        {
            "name": "통신기지국 백업전원", "icon": "📡",
            "desc": "기지국 정전 대비. 부동충전 위주, 연간 수회 방전.",
            "ref": "EverExceed (업계표준); Martinez-Laserna et al. (2018)",
            "score": max(0, adjusted_health - 15),
            "condition": adjusted_health >= 50,
        },
        {
            "name": "전기차 보조 배터리", "icon": "🚗",
            "desc": "저출력 범위. 일일 충방전 100회 이상 가능.",
            "ref": "Circunomics; Frontiers Chemistry",
            "score": max(0, adjusted_health - 20),
            "condition": adjusted_health >= 50,
        },
    ]

    recs_filtered = [a for a in apps if a['condition'] and a['score'] > 0]
    recs_sorted   = sorted(recs_filtered, key=lambda x: x['score'], reverse=True)
    return recs_sorted, adjusted_health, weights['desc']

def safety_eval(health, years, cycles, bat_type, voltage):
    """
    안전성 판정 3단계 (PMC11033388 기준)
    보조 경고: 사이클 수명 (Ali et al. 2023) + 캘린더 열화 + LFP 임피던스
    ※ 보조 경고는 판정 등급을 바꾸지 않고 경고 메시지만 추가
    """
    props       = BAT_PROPS[bat_type]
    cycle_ratio = cycles / props['cycle_life']
    cal_rate    = props.get('calendar_aging_rate_pct', 2.0)
    cal_loss    = years * cal_rate
    tier        = get_soh_tier(health)

    warnings = []
    if cycle_ratio > 1.0:
        warnings.append(
            f"⚠️ 설계 사이클 수명 초과 ({int(cycles)}회 / 기준 {props['cycle_life']}회)"
            f" — 집중 모니터링 권장 (Ali et al. 2023)"
        )
    elif cycle_ratio > 0.75:
        warnings.append(
            f"⚠️ 사이클 수명 {round(cycle_ratio*100)}% 소모 — 주기적 점검 권장 (Ali et al. 2023)"
        )
    if cal_loss >= 10:
        warnings.append(
            f"⚠️ 캘린더 열화 누적 약 {cal_loss:.0f}% 예상"
            f" ({bat_type} {cal_rate}%/년 × {years}년, Ali et al. 2023)"
        )
    if bat_type == "LFP":
        thr = props.get('eis_threshold', 60)
        if health < thr:
            warnings.append(f"⚠️ LFP SOH {thr}% 미만: 임피던스 급증 구간 (PMC11033388)")
    warn_str = (" | " + " | ".join(warnings)) if warnings else ""

    if tier == "recycle":
        return ("위험 — 해체(Recycle)", "#e05555",
                f"SOH ≤ 50%: 재활용 공정 투입 필요 (PMC11033388){warn_str}")
    elif tier == "repurpose":
        return ("주의 — 재활용(Repurpose)", "#f0a500",
                f"SOH 50~80%: 제한된 용도 재활용 가능, 주기적 점검 필요 (PMC11033388){warn_str}")
    else:
        return ("양호 — 재사용(Reuse)", "#00d4aa",
                f"SOH > 80%: 안전한 재사용 가능 (PMC11033388; IEC 62933, UL 1974){warn_str}")

def render_result(soh_final, soh_source, bat_type, years, cycles, voltage, mode_label):
    """진단 결과 공통 렌더링 (SOH 판정 + 안전성 + 추천 활용처 + 최종 요약)"""
    tier           = get_soh_tier(soh_final)
    tier_text, tier_color = TIER_META[tier]
    soh_color      = "#00d4aa" if soh_final > 80 else "#f0a500" if soh_final > 50 else "#e05555"

    # LFP 2차 수명 안내 (PMC11033388 + sdsu.edu 논문)
    if bat_type == "LFP" and 75 <= soh_final <= 85:
        p = BAT_PROPS["LFP"]
        st.info(
            f"🔋 **LFP 2차 수명 안내** (SOH ≈ 80% 전환 시점)\n\n"
            f"전압 2.80~3.55V / 충전 0.5C / 방전 1C 조건에서 "
            f"용량 60% 도달까지 **{p['second_life_cycles'][0]:,}~{p['second_life_cycles'][1]:,}사이클**, "
            f"하루 1회 기준 **{p['second_life_years'][0]}~{p['second_life_years'][1]}년** 기대\n\n"
            f"📚 출처: chrismi.sdsu.edu/publications/225.pdf"
        )

    if bat_type == "LFP":
        thr = BAT_PROPS["LFP"].get('eis_threshold', 60)
        if soh_final < thr:
            st.warning(
                f"⚠️ **LFP 임피던스 주의**: SOH {thr}% 미만 — 임피던스 급증 및 용량 저하 시작 (PMC11033388)"
            )

    st.markdown(
        f'<div class="section-title">🤖 진단 결과 '
        f'<span style="font-size:13px;color:#888;">— {mode_label}</span></div>',
        unsafe_allow_html=True
    )
    m1, m2, m3, m4 = st.columns(4)
    for col, val, label, color, note in zip(
        [m1, m2, m3, m4],
        [f"{soh_final}%", bat_type,
         f"{int(cycles) if float(cycles)==int(float(cycles)) else round(float(cycles))}회 / {round(float(years),1)}년",
         tier_text],
        ["SOH", "배터리 종류", "사이클 / 사용 연수", "판정 등급"],
        [soh_color, "#00d4aa", "#00d4aa", tier_color],
        [soh_source[:28]+"...", "화학 조성", "누적 사용 이력",
         ">80% 재사용 / 50~80% 재활용 / ≤50% 해체"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:{color}">{val}</div>
            <div class="metric-label">{label}</div>
            <div class="ref-text">{note}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🛡️ 안전성 평가</div>', unsafe_allow_html=True)
    s_txt, s_color, s_desc = safety_eval(soh_final, years, cycles, bat_type, voltage)
    st.markdown(f"""
    <div class="metric-card" style="text-align:left; border:2px solid {s_color};">
        <span style="font-size:20px; font-weight:700; color:{s_color}">{s_txt}</span>
        <span style="font-size:14px; color:#ccc; margin-left:12px;">{s_desc}</span>
    </div>""", unsafe_allow_html=True)

    # 조정된 SOH 계산 및 추천
    recs, adjusted_health, battery_desc = get_recommendations(soh_final, years, cycles, bat_type, voltage)
    adjustment_pct = soh_final - adjusted_health
    if adjustment_pct > 0.1:
        st.markdown('<div class="section-title">📊 조정된 SOH 분석</div>', unsafe_allow_html=True)
        st.info(
            f"**{battery_desc}**\n\n"
            f"기본 SOH: {soh_final:.1f}% → 조정된 SOH: **{adjusted_health:.1f}%**\n\n"
            f"*(사이클 {int(cycles)}회, 연수 {years}년, 전압 {voltage}V 고려, 페널티: -{adjustment_pct:.1f}%)*"
        )

    st.markdown('<div class="section-title">🎯 추천 활용처</div>', unsafe_allow_html=True)
    st.caption("📌 활용처별 기준: 조정된 SOH (배터리별 차등 페널티 포함) | Edge et al. (2023); Martinez-Laserna et al. (2018)")

    if s_txt.startswith("위험"):
        st.error("❌ **위험 상태** — 2차 활용처 불가\n\n해체 필요. 배터리를 즉시 재활용 공정에 투입하세요.")
    elif not recs:
        st.error("❌ 모든 활용처 기준 미달 — 재활용 공정 투입 권장 (조정 SOH 50% 미만)")
    else:
        for i, rec in enumerate(recs):
            cls  = "rec-card top-card" if i == 0 else "rec-card"
            rank = "✦ 최우선 추천" if i == 0 else f"{i+1}순위 추천"
            st.markdown(f"""
            <div class="{cls}">
                <div style="font-size:18px; font-weight:900; color:#FFFFFF; margin-bottom:8px;">{rec['icon']} {rec['name']}</div>
                <div style="font-size:13px; color:#00FF88; margin-bottom:8px; font-weight:700;">{rank} · 적합도 {round(rec['score'])}점</div>
                <div style="font-size:14px; color:#DDDDDD; margin-bottom:6px; line-height:1.6;">{rec['desc']}</div>
                <div style="font-size:12px; color:#AAAAAA;">📚 {rec['ref']}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    cycle_pct = round(float(cycles) / BAT_PROPS[bat_type]['cycle_life'] * 100)
    if tier == "recycle":
        fc, fm, fr = "#e05555", "❌ 재사용 불가 — 해체(Recycle) 공정 필요", "PMC11033388 (SOH ≤ 50%); IEC 62619"
    elif tier == "repurpose":
        fc, fm, fr = "#f0a500", "♻️ 재활용(Repurpose) 가능 — 제한된 용도 사용, 주기적 점검 필요", "PMC11033388 (SOH 50~80%)"
    else:
        fc, fm, fr = "#00d4aa", "✅ 재사용(Reuse) 가능", "PMC11033388 (SOH > 80%); IEC 62933, UL 1974"

    st.markdown(f"""
    <div style="background:#1a1a2e; border-radius:12px; padding:20px;
                border:2px solid {fc}; text-align:center;">
        <div style="font-size:24px; font-weight:700; color:{fc}">{fm}</div>
        <div style="font-size:13px; color:#aaa; margin-top:8px;">
            배터리 종류: {bat_type} | SOH: {soh_final}% | 사용 연수: {years}년 |
            충방전: {cycles}회 ({cycle_pct}% 소모) | 전압: {voltage}V
        </div>
        <div style="font-size:11px; color:#666; margin-top:6px;">📚 근거: {fr}</div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# 메인 앱 시작
# ═══════════════════════════════════════════════
st.markdown('<h1 class="main-title">🔋 배터리 Second-Life 추천 플랫폼</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">EIS 또는 BMS 데이터 기반 배터리 상태 진단 및 재사용/재활용/해체 판정</p>',
    unsafe_allow_html=True
)

# 모델 로드
with st.spinner("🤖 모델 로딩 중..."):
    eis_result  = train_eis_model()
    bms_models, bms_cv_gb, bms_cv_rf = train_bms_model()

eis_models = eis_result[0]
eis_n      = eis_result[1]
eis_cv_gb  = eis_result[2]
eis_cv_rf  = eis_result[3]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("EIS 학습 파일",  f"{eis_n}개" if eis_n else "미로드")
c2.metric("EIS GB R²",     f"{eis_cv_gb:.4f}" if eis_cv_gb else "—")
c3.metric("EIS RF R²",     f"{eis_cv_rf:.4f}" if eis_cv_rf else "—")
c4.metric("BMS GB R²",     f"{bms_cv_gb:.4f}")
c5.metric("BMS RF R²",     f"{bms_cv_rf:.4f}")

st.divider()

# 사이드바
with st.sidebar:
    st.markdown("### 📋 배터리 기본 정보")
    bat_type = st.selectbox("배터리 종류", ["LFP", "NCM", "NCA", "LCO"])

    # BMS 기반 예측 모드: 연수·사이클은 파일마다 달라서 사이드바 슬라이더 불필요
    _is_bms_mode = st.session_state.get('current_method', '') == '🏭 BMS 기반 예측'

    if _is_bms_mode:
        st.markdown(
            "<div style=\"background:#111827; border:1px solid #374151; border-radius:8px;"
            "padding:10px 14px; margin-bottom:8px;\">"
            "<span style=\"color:#9ca3af; font-size:12px;\">📂 사용 연수 · 충방전 횟수</span><br>"
            "<span style=\"color:#00d4aa; font-size:13px;\">"
            "파일별로 자동 추출됩니다<br>"
            "<span style=\"color:#6b7280; font-size:11px;\">"
            "각 배터리마다 값이 달라 결과 테이블에서 확인하세요</span>"
            "</span></div>",
            unsafe_allow_html=True
        )
        years  = 0
        cycles = 0
    else:
        years  = st.slider("사용 연수 (년)", 0, 15, 0)
        cycles = st.slider(
            "충방전 횟수", 0, 10000, 0, 100,
            help="NCM 2,000 / LFP 4,000 / NCA 1,500 / LCO 800회가 설계 수명 기준 (Ali et al. 2023)"
        )
    voltage  = st.number_input("현재 전압 (V)", 2.0, 4.3, 3.2, step=0.1)

    st.divider()
    st.markdown("### 🔬 분석 방법 선택")
    method = st.radio(
        "",
        ["⚡ EIS 기반 예측",
         "✏️ SOH 직접 입력",
         "🏭 BMS 기반 예측"],
        help=(
            "EIS: 임피던스 파일 업로드 (정밀) | "
            "BMS: 여러 배터리 .mat/.csv 일괄 분석 → 사이클·연수 자동 추출 → 엑셀 다운로드"
        )
    )

# 현재 모드 저장 (사이드바 조건 분기용)
st.session_state['current_method'] = method

with st.expander("ℹ️ SOH 판정 기준 (PMC11033388)", expanded=False):
    st.markdown("""
| SOH | 판정 | 주요 활용처 |
|---|---|---|
| **> 80%** | ✅ 재사용 | EV, ESS, 고성능 |
| **70~80%** | ♻️ 재활용 | ESS, 그리드 |
| **50~80%** | ♻️ 재활용 | UPS, 통신 백업 (grade C) |
| **≤ 50%** | 🗑️ 해체 | 재활용 공정 |

> **LFP**: SOH 60% 미만부터 임피던스 급증 (PMC11033388)
    """)

# ─────────────────────────────────────────────
# 모드 1: EIS 기반 예측
# ─────────────────────────────────────────────
if method == "⚡ EIS 기반 예측":
    st.markdown("### 📂 EIS 파일 업로드")
    st.caption("Warwick DIB 포맷 (.xls) 또는 freq/z_real/z_imag 3열 CSV | 반복 측정 여러 개 → 자동 평균")
    uploaded = st.file_uploader(
        "EIS 파일 (.xls / .csv)",
        type=['xls', 'xlsx', 'csv'],
        accept_multiple_files=True
    )

    if uploaded:
        all_zr, all_zi, df_list = [], [], []
        for f in uploaded:
            try:
                raw = f.read()
                zr, zi = parse_csv_eis(raw) if f.name.endswith('.csv') else parse_xls_eis(raw)
                all_zr.append(zr); all_zi.append(zi)
                ml = min(len(zr), len(zi)) if zi else len(zr)
                df_list.append(pd.DataFrame({'z_real': zr[:ml],
                                             'z_imag': zi[:ml] if zi else [0]*ml}))
            except Exception as e:
                st.warning(f"⚠️ {f.name} 읽기 실패: {e}")

        if not all_zr:
            st.error("읽을 수 있는 파일이 없습니다.")
            st.stop()

        max_len = max(len(z) for z in all_zr)
        pad     = lambda l, n: l + [l[-1]]*(n-len(l)) if l else [0]*n
        avg_zr  = np.mean([pad(z, max_len) for z in all_zr], axis=0).tolist()
        avg_zi  = np.mean([pad(z, max_len) for z in all_zi], axis=0).tolist() \
                  if all_zi[0] else []

        if len(uploaded) > 1:
            st.caption(f"📊 {len(uploaded)}개 파일 평균값으로 분석")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-title">📈 나이퀴스트 플롯</div>', unsafe_allow_html=True)
            fig = go.Figure()
            for d in df_list:
                fig.add_trace(go.Scatter(x=d['z_real'], y=-d['z_imag'], mode='lines',
                                         line=dict(color='rgba(0,212,170,0.2)', width=1),
                                         showlegend=False))
            ml = min(len(avg_zr), len(avg_zi)) if avg_zi else len(avg_zr)
            fig.add_trace(go.Scatter(
                x=avg_zr[:ml], y=[-v for v in avg_zi[:ml]] if avg_zi else [0]*ml,
                mode='lines+markers', name='평균',
                marker=dict(color=list(range(ml)), colorscale='Plasma', size=7,
                            colorbar=dict(title="포인트", thickness=12)),
                line=dict(color='rgba(255,255,255,0.8)', width=2)
            ))
            fig.update_layout(xaxis_title="Z' (Ω)", yaxis_title="-Z'' (Ω)",
                              template='plotly_dark', height=300,
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">📊 임피던스 크기</div>', unsafe_allow_html=True)
            fig2 = go.Figure()
            avg_zm = np.sqrt(np.array(avg_zr)**2 +
                             np.array(avg_zi if avg_zi else [0]*len(avg_zr))**2) * 1000
            fig2.add_trace(go.Scatter(y=avg_zm, mode='lines+markers',
                                       line=dict(color='#00d4aa', width=2),
                                       marker=dict(size=5)))
            fig2.update_layout(xaxis_title="포인트 (고주파→저주파)",
                               yaxis_title="|Z| (mΩ)",
                               template='plotly_dark', height=300,
                               margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        if eis_models:
            soh = predict_soh_eis(eis_models, avg_zr, avg_zi)
            if soh:
                render_result(soh, "EIS ML 예측 (Warwick DIB, 앙상블)",
                              bat_type, years, cycles, voltage, "⚡ EIS 기반")
            else:
                st.error("EIS 피처 추출 실패. 파일 형식을 확인해주세요.")
        else:
            st.warning("⚠️ EIS 모델 미로드 — Cell02_95SOH_15degC_05SOC_9505.zip 파일을 확인해주세요.")
    else:
        st.info("👆 EIS 파일을 업로드하면 분석이 시작됩니다.")
        st.markdown("""
**측정 조건 권장사항**
- 온도 25°C, SOC 50% 기준 측정 (온도·SOC에 따라 임피던스 값 변동)
- 주파수 범위: 0.01 Hz ~ 100 kHz
- 반복 측정 여러 개 업로드 시 자동 평균 처리
        """)

# ─────────────────────────────────────────────
# 모드 2: SOH 직접 입력
# ─────────────────────────────────────────────
elif method == "✏️ SOH 직접 입력":
    st.markdown("### ✏️ SOH 직접 입력")
    st.caption("실측 용량 데이터 또는 외부 측정 장비 결과를 직접 입력합니다.")
    soh_direct = st.slider(
        "SOH (%)", 10, 100, 80,
        help="SOH = 현재 실제 용량 / 신품 정격 용량 × 100 (IEC 62660-1)"
    )
    if st.button("📊 분석 실행", type="primary"):
        render_result(soh_direct, "직접 입력 (IEC 62660-1 기준)",
                      bat_type, years, cycles, voltage, "✏️ 직접 입력")

# ─────────────────────────────────────────────
# 모드 3: BMS 기반 예측 (기업용)
# ─────────────────────────────────────────────
elif method == "🏭 BMS 기반 예측":
    st.markdown("### 🏭 BMS 기반 예측 — 여러 배터리 일괄 분석")
    st.caption(
        "BMS 데이터 파일 여러 개 업로드 → 사이클·연수 자동 추출 → SOH 계산 → "
        "재사용/재활용/해체 판정 → 엑셀 다운로드"
    )

    batch_mode = st.radio(
        "파일 형식",
        ["📂 .mat 파일 (NASA PCoE MATLAB 형식)", "📁 CSV 파일 (용량 실측 데이터)"],
        horizontal=True
    )

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        batch_bat_type = st.selectbox(
            "배터리 종류 (전체 공통 적용)",
            ["LFP", "NCM", "NCA", "LCO"],
            help="모든 파일에 동일하게 적용됩니다."
        )
    with col_opt2:
        batch_voltage = st.number_input(
            "공칭 전압 (V)", 2.0, 4.3,
            float(BAT_PROPS[batch_bat_type]['nominal_v']), step=0.1
        )

    if batch_mode == "📂 .mat 파일 (NASA PCoE MATLAB 형식)":
        st.caption("각 파일 = 배터리 1개. 최신 사이클 기준 방전 용량으로 SOH 자동 계산.")
        batch_files = st.file_uploader(
            ".mat 파일 업로드 (여러 개 동시 가능)",
            type=['mat'], accept_multiple_files=True
        )
    else:
        st.caption("각 파일 = 배터리 1개. capacity 컬럼 있으면 실측 SOH, 없으면 ML 추정.")
        with st.expander("📄 CSV 컬럼 형식 안내"):
            st.markdown("""
| 컬럼명 | 단위 | 설명 |
|---|---|---|
| `cycle` | 정수 | 사이클 번호 |
| `capacity` | Ah | 방전 용량 (있으면 실측 SOH 계산) |
| `voltage` | V | 전압 |
| `current` | A | 전류 |
| `temperature` | °C | 온도 |
| `time_s` | s | 시간 |

> capacity 컬럼이 없으면 ML 모델로 SOH 추정합니다.
            """)
            sample = pd.DataFrame({
                'cycle':       [1, 1, 1, 2, 2, 2],
                'voltage':     [3.2, 3.18, 3.15, 3.19, 3.17, 3.14],
                'current':     [1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
                'temperature': [25.1, 25.3, 25.5, 25.2, 25.4, 25.6],
                'capacity':    [0.0, 0.5, 2.0, 0.0, 0.49, 1.97],
                'time_s':      [0, 1800, 3600, 0, 1800, 3600],
            })
            st.download_button("⬇️ 샘플 CSV 다운로드", sample.to_csv(index=False),
                               "bms_sample.csv", "text/csv")
        batch_files = st.file_uploader(
            "CSV 파일 업로드 (여러 개 동시 가능)",
            type=['csv'], accept_multiple_files=True
        )

    if batch_files:
        st.divider()
        st.markdown(f"**{len(batch_files)}개 파일 분석 중...**")
        results, errors = [], []
        progress = st.progress(0)
        status   = st.empty()

        for i, f in enumerate(batch_files):
            status.text(f"처리 중: {f.name} ({i+1}/{len(batch_files)})")
            progress.progress((i+1) / len(batch_files))
            try:
                raw = f.read()
                if batch_mode == "📂 .mat 파일 (NASA PCoE MATLAB 형식)":
                    cycles_data, bat_name = parse_nasa_mat(raw)
                    soh_records = compute_soh_from_mat(cycles_data)
                    if not soh_records:
                        errors.append({'파일명': f.name, '오류': '방전 사이클 없음'}); continue
                    last       = soh_records[-1]
                    soh_val    = round(last['soh_actual'], 2)
                    soh_method = "실측 (방전 용량)"
                    cycle_cnt  = last['discharge_idx']
                    est_years  = round(cycle_cnt / 365.0, 1)
                    capacity   = round(last['capacity_ah'], 4)

                else:
                    df_b = pd.read_csv(io.BytesIO(raw))
                    col_map = {}
                    for col in df_b.columns:
                        cl_name = col.lower()
                        if any(k in cl_name for k in ['cycle','cyc','사이클']):     col_map['cycle']       = col
                        elif any(k in cl_name for k in ['voltage','volt','전압']):  col_map['voltage']     = col
                        elif any(k in cl_name for k in ['current','amp','전류']):   col_map['current']     = col
                        elif any(k in cl_name for k in ['temp','온도']):            col_map['temperature'] = col
                        elif any(k in cl_name for k in ['time','시간']):            col_map['time_s']      = col
                        elif any(k in cl_name for k in ['capacity','용량','cap']):  col_map['capacity']    = col
                    df_b = df_b.rename(columns={v: k for k, v in col_map.items()})

                    if 'capacity' in df_b.columns and 'cycle' in df_b.columns:
                        cap_per = df_b.groupby('cycle')['capacity'].max()
                        rated   = cap_per.iloc[0]
                        last_cap = cap_per.iloc[-1]
                        soh_val  = round(last_cap / rated * 100, 2)
                        soh_method = "실측 (방전 용량)"
                        cycle_cnt  = int(cap_per.index[-1])
                        est_years  = round(cycle_cnt / 365.0, 1)
                        capacity   = round(float(last_cap), 4)
                    elif {'cycle', 'voltage', 'current', 'temperature', 'time_s'}.issubset(df_b.columns):
                        # capacity 없을 때 ML 추정: 사이클 수 + 연수 + 내부저항 기반
                        last_cycle = df_b['cycle'].max()
                        est_years  = round(float(last_cycle) / 365.0, 1)
                        r_ref_b    = BAT_RESISTANCE[batch_bat_type]
                        cr_ratio_b = min(last_cycle / BAT_PROPS[batch_bat_type]['cycle_life'], 1.3)
                        auto_r_b   = r_ref_b['r0'] + (r_ref_b['r_max'] - r_ref_b['r0']) * cr_ratio_b
                        soh_val    = predict_soh_bms(bms_models, batch_bat_type,
                                                     last_cycle, est_years, auto_r_b)
                        soh_method = "ML 추정 (용량 미측정)"
                        cycle_cnt  = int(last_cycle)
                        capacity   = None
                    else:
                        errors.append({'파일명': f.name, '오류': f"필수 컬럼 없음: {list(df_b.columns)}"}); continue

                tier        = get_soh_tier(soh_val)
                tier_text, _ = TIER_META[tier]
                props_b     = BAT_PROPS[batch_bat_type]
                recs, adjusted_h, _ = get_recommendations(soh_val, est_years, cycle_cnt,
                                                           batch_bat_type, batch_voltage)
                top_rec     = recs[0]['name'] if recs else "해당 없음 (해체 권장)"
                cycle_ratio = cycle_cnt / props_b['cycle_life']
                cal_loss    = est_years * props_b.get('calendar_aging_rate_pct', 2.0)

                results.append({
                    '파일명':           f.name,
                    '배터리 종류':      batch_bat_type,
                    'SOH (%)':          soh_val,
                    'SOH 계산 방법':    soh_method,
                    '판정':             tier_text,
                    '1순위 활용처':     top_rec,
                    '방전 사이클':      cycle_cnt,
                    '추정 연수':        est_years,
                    '사이클 소모율 (%)': round(cycle_ratio * 100, 1),
                    '캘린더 열화 예상 (%)': round(cal_loss, 1),
                    '측정 용량 (Ah)':   capacity if capacity else '-',
                })
            except Exception as e:
                errors.append({'파일명': f.name, '오류': str(e)})

        progress.empty(); status.empty()

        if results:
            df_result = pd.DataFrame(results)
            st.markdown('<div class="section-title">📊 분석 결과 요약</div>', unsafe_allow_html=True)
            total      = len(df_result)
            reuse      = len(df_result[df_result['판정'].str.contains('재사용')])
            repurpose  = len(df_result[df_result['판정'].str.contains('재활용')])
            recycle    = len(df_result[df_result['판정'].str.contains('해체')])
            c1,c2,c3,c4 = st.columns(4)
            for col, val, label, color in zip(
                [c1, c2, c3, c4],
                [total, reuse, repurpose, recycle],
                ["총 배터리 수", "✅ 재사용", "♻️ 재활용", "🗑️ 해체"],
                ["#00d4aa", "#00d4aa", "#f0a500", "#e05555"]
            ):
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color:{color}">{val}개</div>
                    <div class="metric-label">{label}</div>
                    <div class="ref-text">{round(val/total*100)}%</div>
                </div>""", unsafe_allow_html=True)

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df_result['SOH (%)'], nbinsx=20,
                marker_color='#00d4aa', opacity=0.8, name='배터리 수'
            ))
            fig_hist.add_vline(x=80, line_dash='dash', line_color='#f0a500',
                               annotation_text='80% 재사용 기준')
            fig_hist.add_vline(x=50, line_dash='dash', line_color='#e05555',
                               annotation_text='50% 해체 기준')
            fig_hist.update_layout(
                xaxis_title='SOH (%)', yaxis_title='배터리 수 (개)',
                template='plotly_dark', height=300,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown('<div class="section-title">📋 배터리별 상세 결과</div>', unsafe_allow_html=True)
            st.dataframe(df_result, use_container_width=True, hide_index=True)

            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                df_result.to_excel(writer, sheet_name='배터리_SOH_분석', index=False)
                summary = pd.DataFrame({
                    '구분':   ['총 배터리', '재사용 (SOH>80%)', '재활용 (SOH 50~80%)', '해체 (SOH≤50%)'],
                    '수량':   [total, reuse, repurpose, recycle],
                    '비율(%)': [100, round(reuse/total*100,1),
                                round(repurpose/total*100,1), round(recycle/total*100,1)],
                })
                summary.to_excel(writer, sheet_name='요약', index=False)

            st.download_button(
                label="⬇️ 엑셀 다운로드 (.xlsx)",
                data=excel_buf.getvalue(),
                file_name="배터리_SOH_분석결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

        if errors:
            with st.expander(f"⚠️ 처리 실패 파일 {len(errors)}개", expanded=False):
                st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)
    else:
        st.info(
            "👆 .mat 또는 CSV 파일을 업로드하면 분석이 시작됩니다.\n"
            ".mat: 사이클수·연수 자동 추출 후 사이드바에 반영 | "
            "CSV: capacity 컬럼 있으면 실측 SOH 계산\n"
            "결과: SOH, 재사용/재활용/해체 판정, 추천 활용처 → 엑셀 다운로드"
        )
