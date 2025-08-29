# -*- coding: utf-8 -*-
# app.py — ARAM 챔피언 대시보드 + 생성형 AI Copilot (아이콘: 챔피언/아이템/스펠/룬)
# - 사이드바에서 OpenAI API 키 입력 가능(세션에만 저장). 없으면 env/Secrets 사용.
# - 탭: [Dashboard] 기존 분석, [AI Copilot] 생성형 챗봇(대시보드 컨텍스트 주입)
# - 패키지: streamlit, pandas, openai>=1.0,<2, python-dotenv(선택)

import os, re
import pandas as pd
import streamlit as st

# --- (선택) .env 지원: 로컬에서 쉽게 쓰려면 .env에 OPENAI_API_KEY 넣어두면 자동 인식 ---
try:
    from dotenv import load_dotenv  # optional
    load_dotenv()
except Exception:
    pass

# OpenAI SDK (v1)
try:
    from openai import OpenAI
    _openai_ok = True
except Exception:
    _openai_ok = False

st.set_page_config(page_title="ARAM PS Dashboard + AI Copilot", layout="wide")

# ===== 파일 경로(리포 루트) =====
PLAYERS_CSV   = "aram_participants_with_icons_superlight.csv"  # 참가자 행 데이터
ITEM_SUM_CSV  = "item_summary_with_icons.csv"                  # item, icon_url, total_picks, wins, win_rate
CHAMP_CSV     = "champion_icons.csv"                           # champion, champion_icon (또는 icon/icon_url)
RUNE_CSV      = "rune_icons.csv"                               # rune_core, rune_core_icon, rune_sub, rune_sub_icon
SPELL_CSV     = "spell_icons.csv"                              # 스펠이름, 아이콘URL (헤더 자유)

DD_VERSION = "15.16.1"  # Data Dragon 폴백 버전 (필요시 최신으로 교체)

# ===== 유틸 =====
def _exists(path: str) -> bool:
    ok = os.path.exists(path)
    if not ok:
        st.warning(f"파일 없음: `{path}`")
    return ok

def _norm(x: str) -> str:
    return re.sub(r"\s+", "", str(x)).strip().lower()

# ===== 로더 =====
@st.cache_data
def load_players(path: str) -> pd.DataFrame:
    if not _exists(path):
        st.stop()
    df = pd.read_csv(path, encoding='utf-8')

    # 승패 정리
    if "win_clean" not in df.columns:
        if "win" in df.columns:
            df["win_clean"] = df["win"].astype(str).str.lower().isin(["true","1","t","yes"]).astype(int)
        else:
            df["win_clean"] = 0

    # 아이템 이름 정리
    for c in [c for c in df.columns if re.fullmatch(r"item[0-6]_name", c)]:
        df[c] = df[c].fillna("").astype(str).str.strip()

    # 기본 텍스트 컬럼
    for c in ["spell1","spell2","spell1_name_fix","spell2_name_fix","rune_core","rune_sub","champion"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()
    return df

@st.cache_data
def load_item_summary(path: str) -> pd.DataFrame:
    if not _exists(path):
        return pd.DataFrame()
    g = pd.read_csv(path)
    need = {"item","icon_url","total_picks","wins","win_rate"}
    if not need.issubset(g.columns):
        st.warning(f"`{path}` 헤더 확인 필요 (기대: {sorted(need)}, 실제: {list(g.columns)})")
    if "item" in g.columns:
        g = g[g["item"].astype(str).str.strip() != ""]
    return g

@st.cache_data
def load_champion_icons(path: str) -> dict:
    if not _exists(path):
        return {}
    df = pd.read_csv(path)
    name_col = next((c for c in ["champion","Champion","championName"] if c in df.columns), None)
    icon_col = next((c for c in ["champion_icon","icon","icon_url"] if c in df.columns), None)
    if not name_col or not icon_col:
        return {}
    df[name_col] = df[name_col].astype(str).str.strip()
    return dict(zip(df[name_col], df[icon_col]))

@st.cache_data
def load_rune_icons(path: str) -> dict:
    if not _exists(path):
        return {"core": {}, "sub": {}, "shards": {}}
    df = pd.read_csv(path)
    core_map, sub_map, shard_map = {}, {}, {}
    if "rune_core" in df.columns:
        ic = "rune_core_icon" if "rune_core_icon" in df.columns else None
        if ic: core_map = dict(zip(df["rune_core"].astype(str), df[ic].astype(str)))
    if "rune_sub" in df.columns:
        ic = "rune_sub_icon" if "rune_sub_icon" in df.columns else None
        if ic: sub_map = dict(zip(df["rune_sub"].astype(str), df[ic].astype(str)))
    if "rune_shard" in df.columns:
        ic = "rune_shard_icon" if "rune_shard_icon" in df.columns else ("rune_shards_icons" if "rune_shards_icons" in df.columns else None)
        if ic: shard_map = dict(zip(df["rune_shard"].astype(str), df[ic].astype(str)))
    return {"core": core_map, "sub": sub_map, "shards": shard_map}

@st.cache_data
def load_spell_icons(path: str) -> dict:
    """스펠명(여러 형태) -> 아이콘 URL"""
    if not _exists(path):
        return {}
    df = pd.read_csv(path)
    cand_name = [c for c in df.columns if _norm(c) in {"spell","spellname","name","spell1_name_fix","spell2_name_fix","스펠","스펠명"}]
    cand_icon = [c for c in df.columns if _norm(c) in {"icon","icon_url","spelli con","spell_icon"} or "icon" in c.lower()]
    m = {}
    if cand_name and cand_icon:
        name_col, icon_col = cand_name[0], cand_icon[0]
        for n, i in zip(df[name_col].astype(str), df[icon_col].astype(str)):
            m[_norm(n)] = i
            m[str(n).strip()] = i
    else:
        if df.shape[1] >= 2:
            for n, i in zip(df.iloc[:,0].astype(str), df.iloc[:,1].astype(str)):
                m[_norm(n)] = i
                m[str(n).strip()] = i
    return m

# ===== 데이터 로드 =====
df        = load_players(PLAYERS_CSV)
item_sum  = load_item_summary(ITEM_SUM_CSV)
champ_map = load_champion_icons(CHAMP_CSV)
rune_maps = load_rune_icons(RUNE_CSV)
spell_map = load_spell_icons(SPELL_CSV)

ITEM_ICON_MAP = dict(zip(item_sum.get("item", []), item_sum.get("icon_url", [])))

# ===== 사이드바: 공통 컨트롤 + OpenAI 설정 =====
st.sidebar.title("ARAM PS Controls")

champs = sorted(df["champion"].dropna().unique().tolist()) if "champion" in df.columns else []
selected = st.sidebar.selectbox("Champion", champs, index=0 if champs else None)

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 OpenAI 설정")

# 1) 우선순위: 세션 ▶️ secrets ▶️ 환경변수
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

api_from_secrets = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
api_from_env     = os.getenv("OPENAI_API_KEY")

api_key_ui = st.sidebar.text_input(
    "OPENAI_API_KEY (여기에 붙여넣기 가능)",
    value=st.session_state.openai_api_key or "",
    type="password",
    placeholder="sk-...",
    help="입력 시 이 세션에서만 사용합니다. 미입력 시 st.secrets 또는 환경변수를 탐색합니다.",
)
if api_key_ui:
    st.session_state.openai_api_key = api_key_ui.strip()

def resolve_api_key() -> str:
    return (
        (st.session_state.openai_api_key or "").strip()
        or (api_from_secrets or "")
        or (api_from_env or "")
    )

model = st.sidebar.selectbox("모델", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
temperature = st.sidebar.slider("창의성(temperature)", 0.0, 1.5, 0.7, 0.1)
max_tokens  = st.sidebar.slider("응답 최대 토큰", 64, 4096, 1024, 64)

st.sidebar.markdown(
    "<small>키 저장: UI 입력(세션 한정) ▶ st.secrets ▶ 환경변수 순</small>",
    unsafe_allow_html=True,
)

# ===== 공용 요약/컨텍스트 계산 함수 (Dashboard/Chat에서 모두 사용) =====
def pick_spell_cols(df_):
    if {"spell1_name_fix","spell2_name_fix"}.issubset(df_.columns):
        return "spell1_name_fix", "spell2_name_fix"
    if {"spell1","spell2"}.issubset(df_.columns):
        return "spell1", "spell2"
    cands = [c for c in df_.columns if "spell" in c.lower()]
    return (cands[0], cands[1]) if len(cands) >= 2 else (None, None)

SPELL_ALIASES = {
    "점멸":"점멸","표식":"표식","눈덩이":"표식","유체화":"유체화","회복":"회복","점화":"점화",
    "정화":"정화","탈진":"탈진","방어막":"방어막","총명":"총명","순간이동":"순간이동",
    "flash":"점멸","mark":"표식","snowball":"표식","ghost":"유체화","haste":"유체화",
    "heal":"회복","ignite":"점화","cleanse":"정화","exhaust":"탈진","barrier":"방어막",
    "clarity":"총명","teleport":"순간이동",
}
KOR_TO_DDRAGON = {
    "점멸":"SummonerFlash","표식":"SummonerSnowball","유체화":"SummonerHaste","회복":"SummonerHeal",
    "점화":"SummonerDot","정화":"SummonerBoost","탈진":"SummonerExhaust","방어막":"SummonerBarrier",
    "총명":"SummonerMana","순간이동":"SummonerTeleport",
}

def standard_korean_spell(s: str) -> str:
    n = _norm(s)
    return SPELL_ALIASES.get(n, s)

def ddragon_spell_icon(s: str) -> str:
    kor = standard_korean_spell(s)
    key = KOR_TO_DDRAGON.get(kor)
    if not key:
        return ""
    return f"https://ddragon.leagueoflegends.com/cdn/{DD_VERSION}/img/spell/{key}.png"

def resolve_spell_icon(name: str) -> str:
    if not name:
        return ""
    raw = str(name).strip()
    for k in (raw, _norm(raw), standard_korean_spell(raw), _norm(standard_korean_spell(raw))):
        if k in spell_map:
            return spell_map[k]
    return ddragon_spell_icon(raw)

def compute_context(selected_champ: str) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """선택 챔피언 요약 텍스트 + top items/spells/runes DataFrame 반환"""
    if not selected_champ or "champion" not in df.columns:
        return "선택된 챔피언이 없습니다.", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    dsel = df[df["champion"] == selected_champ].copy()
    games = len(dsel)
    match_cnt_all = df["matchId"].nunique() if "matchId" in df.columns else len(df)
    match_cnt_sel = dsel["matchId"].nunique() if "matchId" in dsel.columns else games
    winrate = round(dsel["win_clean"].mean()*100, 2) if games else 0.0
    pickrate = round((match_cnt_sel / match_cnt_all * 100), 2) if match_cnt_all else 0.0

    # items
    top_items = pd.DataFrame()
    if games and any(re.fullmatch(r"item[0-6]_name", c) for c in dsel.columns):
        stacks = []
        for c in [c for c in dsel.columns if re.fullmatch(r"item[0-6]_name", c)]:
            stacks.append(dsel[[c, "win_clean"]].rename(columns={c: "item"}))
        union = pd.concat(stacks, ignore_index=True)
        union = union[union["item"].astype(str).str.strip() != ""]
        top_items = (
            union.groupby("item")
            .agg(total_picks=("item","count"), wins=("win_clean","sum"))
            .reset_index()
        )
        top_items["win_rate"] = (top_items["wins"]/top_items["total_picks"]*100).round(2)
        top_items["icon_url"] = top_items["item"].map(ITEM_ICON_MAP)
        top_items = top_items.sort_values(["total_picks","win_rate"], ascending=[False, False]).head(5)

    # spells
    top_spells = pd.DataFrame()
    s1, s2 = pick_spell_cols(dsel)
    if games and s1 and s2:
        sp = (
            dsel.groupby([s1, s2])
            .agg(games=("win_clean","count"), wins=("win_clean","sum"))
            .reset_index()
        )
        sp["win_rate"] = (sp["wins"]/sp["games"]*100).round(2)
        top_spells = sp.sort_values(["games","win_rate"], ascending=[False,False]).head(5)

    # runes
    top_runes = pd.DataFrame()
    if games and {"rune_core","rune_sub"}.issubset(dsel.columns):
        ru = (
            dsel.groupby(["rune_core","rune_sub"])
            .agg(games=("win_clean","count"), wins=("win_clean","sum"))
            .reset_index()
        )
        ru["win_rate"] = (ru["wins"]/ru["games"]*100).round(2)
        top_runes = ru.sort_values(["games","win_rate"], ascending=[False,False]).head(5)

    ctx_lines = [
        f"선택 챔피언: {selected_champ}",
        f"표본 게임수: {games}",
        f"승률: {winrate}%",
        f"픽률: {pickrate}%",
    ]
    return "\n".join(ctx_lines), top_items, top_spells, top_runes

# ===== 탭 구성 =====
tab_dash, tab_ai = st.tabs(["📊 Dashboard", "🤖 AI Copilot"])

with tab_dash:
    # ===== 상단 요약 =====
    dsel = df[df["champion"] == selected].copy() if selected else df.head(0).copy()
    games = len(dsel)
    match_cnt_all = df["matchId"].nunique() if "matchId" in df.columns else len(df)
    match_cnt_sel = dsel["matchId"].nunique() if "matchId" in dsel.columns else games
    winrate = round(dsel["win_clean"].mean()*100, 2) if games else 0.0
    pickrate = round((match_cnt_sel / match_cnt_all * 100), 2) if match_cnt_all else 0.0

    c0, ctitle = st.columns([1, 5])
    with c0:
        cicon = champ_map.get(selected, "")
        if cicon:
            st.image(cicon, width=64)
    with ctitle:
        st.title(f"{selected or 'No champion'}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Games", f"{games}")
    c2.metric("Win Rate", f"{winrate}%")
    c3.metric("Pick Rate", f"{pickrate}%")

    # ===== 아이템 추천 =====
    st.subheader("Recommended Items")
    if games and any(re.fullmatch(r"item[0-6]_name", c) for c in dsel.columns):
        stacks = []
        for c in [c for c in dsel.columns if re.fullmatch(r"item[0-6]_name", c)]:
            stacks.append(dsel[[c, "win_clean"]].rename(columns={c: "item"}))
        union = pd.concat(stacks, ignore_index=True)
        union = union[union["item"].astype(str).str.strip() != ""]
        top_items = (
            union.groupby("item")
            .agg(total_picks=("item","count"), wins=("win_clean","sum"))
            .reset_index()
        )
        top_items["win_rate"] = (top_items["wins"]/top_items["total_picks"]*100).round(2)
        top_items["icon_url"] = top_items["item"].map(ITEM_ICON_MAP)
        top_items = top_items.sort_values(["total_picks","win_rate"], ascending=[False, False]).head(20)

        st.dataframe(
            top_items[["icon_url","item","total_picks","wins","win_rate"]],
            use_container_width=True,
            column_config={
                "icon_url": st.column_config.ImageColumn("아이콘", width="small"),
                "item": "아이템", "total_picks": "픽수", "wins": "승수", "win_rate": "승률(%)"
            }
        )
    else:
        st.info("아이템 이름 컬럼(item0_name~item6_name)이 없어 챔피언별 아이템 집계를 만들 수 없습니다.")

    # ===== 스펠 추천 =====
    st.subheader("Recommended Spell Combos")
    s1, s2 = pick_spell_cols(dsel)
    if games and s1 and s2:
        sp = (
            dsel.groupby([s1, s2])
            .agg(games=("win_clean","count"), wins=("win_clean","sum"))
            .reset_index()
        )
        sp["win_rate"] = (sp["wins"]/sp["games"]*100).round(2)
        sp = sp.sort_values(["games","win_rate"], ascending=[False,False]).head(10)
        sp["spell1_icon"] = sp[s1].apply(resolve_spell_icon)
        sp["spell2_icon"] = sp[s2].apply(resolve_spell_icon)

        st.dataframe(
            sp[["spell1_icon", s1, "spell2_icon", s2, "games", "wins", "win_rate"]],
            use_container_width=True,
            column_config={
                "spell1_icon": st.column_config.ImageColumn("스펠1", width="small"),
                "spell2_icon": st.column_config.ImageColumn("스펠2", width="small"),
                s1: "스펠1 이름", s2: "스펠2 이름",
                "games":"게임수","wins":"승수","win_rate":"승률(%)"
            }
        )
    else:
        st.info("스펠 컬럼을 찾지 못했습니다. (spell1_name_fix/spell2_name_fix 또는 spell1/spell2 필요)")

    # ===== 룬 추천 =====
    st.subheader("Recommended Rune Combos")
    core_map = rune_maps.get("core", {})
    sub_map  = rune_maps.get("sub", {})

    def _rune_core_icon(name: str) -> str: return core_map.get(name, "")
    def _rune_sub_icon(name: str)  -> str: return sub_map.get(name, "")

    if games and {"rune_core","rune_sub"}.issubset(dsel.columns):
        ru = (
            dsel.groupby(["rune_core","rune_sub"])
            .agg(games=("win_clean","count"), wins=("win_clean","sum"))
            .reset_index()
        )
        ru["win_rate"] = (ru["wins"]/ru["games"]*100).round(2)
        ru = ru.sort_values(["games","win_rate"], ascending=[False,False]).head(10)
        ru["rune_core_icon"] = ru["rune_core"].apply(_rune_core_icon)
        ru["rune_sub_icon"]  = ru["rune_sub"].apply(_rune_sub_icon)

        st.dataframe(
            ru[["rune_core_icon","rune_core","rune_sub_icon","rune_sub","games","wins","win_rate"]],
            use_container_width=True,
            column_config={
                "rune_core_icon": st.column_config.ImageColumn("핵심룬", width="small"),
                "rune_sub_icon":  st.column_config.ImageColumn("보조트리", width="small"),
                "rune_core":"핵심룬 이름","rune_sub":"보조트리 이름",
                "games":"게임수","wins":"승수","win_rate":"승률(%)"
            }
        )
    else:
        st.info("룬 컬럼(rune_core, rune_sub)이 없습니다.")

    # ===== 원본(선택 챔피언) =====
    with st.expander("Raw rows (selected champion)"):
        st.dataframe(dsel, use_container_width=True)

with tab_ai:
    st.header("🤖 AI Copilot (생성형 챗봇)")

    if not _openai_ok:
        st.error("`openai` 패키지가 설치되어 있지 않습니다. 터미널에서 `pip install openai` 후 다시 실행하세요.")
        st.stop()

    api_key = resolve_api_key()
    if not api_key:
        st.info("사이드바의 **OPENAI_API_KEY** 입력란에 키를 붙여넣거나, 환경변수/Secrets로 제공하세요.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # 컨텍스트(현재 선택 챔피언 요약) 계산
    ctx_text, ctx_items, ctx_spells, ctx_runes = compute_context(selected)
    with st.expander("현재 대시보드 컨텍스트", expanded=False):
        st.write(ctx_text)
        if not ctx_items.empty:
            st.markdown("**Top Items**")
            st.dataframe(ctx_items, use_container_width=True)
        if not ctx_spells.empty:
            st.markdown("**Top Spells**")
            st.dataframe(ctx_spells, use_container_width=True)
        if not ctx_runes.empty:
            st.markdown("**Top Runes**")
            st.dataframe(ctx_runes, use_container_width=True)

    # 세션 대화 저장소
    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = []

    # 과거 대화 표시
    for m in st.session_state.chat_msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # 입력창
    user_q = st.chat_input("무엇이든 물어보세요 (예: 스펠 추천 이유, 아이템 코어 빌드 등)")
    if user_q:
        st.session_state.chat_msgs.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # 스트리밍 응답
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_text = ""
            try:
                # 시스템 프롬프트: 대시보드 컨텍스트를 활용하도록 지시
                system_prompt = (
                    "You are an ARAM data analyst. Answer in Korean, briefly and clearly. "
                    "When helpful, use the dashboard context provided below.\n\n"
                    f"[DASHBOARD CONTEXT]\n{ctx_text}\n"
                    "- If the user asks about this champion, use the above stats as ground truth.\n"
                    "- If something is unknown, say so honestly."
                )
                # 최근 N개 대화만
                N = 16
                history = st.session_state.chat_msgs[-N:]

                stream = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system_prompt}] + history,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        full_text += delta.content
                        placeholder.markdown(full_text)
            except Exception as e:
                placeholder.error(f"오류: {type(e).__name__} — {e}")

        st.session_state.chat_msgs.append({"role": "assistant", "content": full_text})

    # 수동 내보내기
    if st.session_state.get("chat_msgs"):
        st.download_button(
            label="⬇️ 대화 내보내기 (markdown)",
            data="\n\n".join([f"**{m['role']}**: {m['content']}" for m in st.session_state.chat_msgs]),
            file_name="ai_copilot_chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # 초기화
    if st.button("🧹 Copilot 대화 초기화"):
        st.session_state.chat_msgs = []
        st.rerun()
