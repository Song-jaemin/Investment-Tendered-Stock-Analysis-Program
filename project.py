import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import pykrx.stock
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import plotly.graph_objects as go

# 1. 차트 및 데이터 수집 관련 함수

def plot_candlestick(ticker_code, ticker_name):
    """
    선택된 종목의 1년치 주가 데이터를 가져와 캔들스틱 차트와 이동평균선을 그리는 함수
    Args:
        ticker_code: 종목 코드
        ticker_name: 종목 이름
    """
    try:
        # yfinance를 이용해 1년치 데이터 다운로드
        df = yf.download(ticker_code, period="1y", progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 이동평균선 계산 (20일: 단기 추세, 60일: 중기 추세)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()

        # Plotly를 이용한 인터랙티브 차트 생성
        fig = go.Figure()
        
        # 캔들스틱 차트 추가 (시가, 고가, 저가, 종가)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='주가'))
        
        # 이동평균선 추가 (주황색: 20일선, 파란색: 60일선)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='20일선'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1), name='60일선'))
        
        # 차트 레이아웃 설정
        fig.update_layout(title=f"{ticker_name} 상세 분석 (1년)", height=500, xaxis_rangeslider_visible=False)
        return fig
    except:
        return None

@st.cache_data
def get_fundamental_data():
    """
    pykrx 라이브러리를 사용하여 KOSPI/KOSDAQ 전 종목의 최신 재무 지표(PER, PBR, 배당수익률) 데이터 수집.
    * 주말이나 공휴일에는 데이터가 없으므로 최근 10일간의 데이터를 역추적하여 가장 최신 데이터를 가져옴.
    """
    today = datetime.now()
    
    # 최근 10일을 역순으로 확인하며 데이터가 존재하는 날짜를 탐색
    for days_back in range(0, 11):
        target_date = (today - timedelta(days=days_back)).strftime("%Y%m%d")
        try:
            # 해당 날짜의 시장별 펀더멘털 데이터 수집
            df_kospi = pykrx.stock.get_market_fundamental_by_ticker(target_date, market="KOSPI")
            df_kosdaq = pykrx.stock.get_market_fundamental_by_ticker(target_date, market="KOSDAQ")
            
            # 두 시장 모두 데이터가 없으면(휴장일 등) 이전 날짜로 이동
            if df_kospi.empty and df_kosdaq.empty:
                continue

            # 데이터 합치기
            df_fundamental = pd.concat([df_kospi, df_kosdaq])
            
            # 필요한 컬럼(PER, PBR, DIV)만 추출
            cols = ['PER', 'PBR', 'DIV']
            available_cols = [c for c in cols if c in df_fundamental.columns]
            
            if not df_fundamental.empty:
                df_result = df_fundamental[available_cols].copy()
                df_result.index = df_result.index.astype(str)
                return df_result
        except:
            continue
    return None

# 2. 페이지 설정 및 초기화

# 사이드바를 기본적으로 숨김 상태(collapsed)로 시작
st.set_page_config(layout="wide", page_title="투자 성향 맞춤 분석기", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
</style>
""", unsafe_allow_html=True)

# Streamlit 세션 상태 초기화
if 'step' not in st.session_state:
    st.session_state.step = 1       # 현재 단계 (1: 성향파악, 2: 종목선택, 3: 결과)
if 'propensity' not in st.session_state:
    st.session_state.propensity = None # 투자 성향 결과
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = [] # 선택된 종목 리스트
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None # 분석 결과 데이터
if 'stock_data_prices' not in st.session_state:
    st.session_state.stock_data_prices = None # 주가 데이터
if 'fundamental_data' not in st.session_state:
    st.session_state.fundamental_data = None # 재무 데이터

# 3. 정적 데이터 (섹터 정보, 종목 매핑)

# 관심 분야별 종목 리스트
sectors = {
    'IT/플랫폼': ['005930.KS', '035420.KS', '035720.KS', '032640.KS', '017670.KS', '030200.KS', '005935.KS', '066570.KS'], 
    '반도체': ['000660.KS', '000990.KS', '009150.KS', '011070.KS', '373220.KS', '006400.KS', '003670.KS', '051910.KS'], 
    '바이오/헬스케어': ['207940.KS', '068270.KS', '328150.KQ', '178920.KQ', '000100.KS', '128940.KS', '009290.KS', '008930.KS'], 
    '금융': ['105560.KS', '055550.KS', '323410.KS', '000810.KS', '034730.KS', '006800.KS', '016360.KS', '039490.KS'], 
    '자동차': ['005380.KS', '000270.KS', '012330.KS', '018880.KS', '011790.KS', '010130.KS', '004020.KS', '017940.KS'], 
    '배터리/화학': ['373220.KS', '051910.KS', '096770.KS', '086520.KQ', '006400.KS', '003670.KS', '000150.KS', '004990.KS'], 
    '엔터테인먼트': ['352820.KS', '035900.KS', '041510.KQ', '035760.KQ', '140410.KQ', '131970.KQ', '237880.KQ', '187790.KQ'], 
    '게임': ['251270.KS', '036570.KS', '112040.KQ', '293490.KQ', '078600.KQ', '259960.KQ', '052790.KQ', '042040.KQ']  
}

# 티커를 종목명으로 변환하기 위한 맵
ticker_name_map = {
    '005930.KS': '삼성전자', '035420.KS': 'NAVER', '035720.KS': '카카오', '032640.KS': 'LG유플러스', '017670.KS': 'SK텔레콤', '030200.KS': 'KT', '005935.KS': '삼성전자우', '066570.KS': 'LG전자',
    '000660.KS': 'SK하이닉스', '000990.KS': 'DB하이텍', '009150.KS': '삼성전기', '011070.KS': 'LG이노텍', '373220.KS': 'LG에너지솔루션', '006400.KS': '삼성SDI', '003670.KS': '포스코퓨처엠', '051910.KS': 'LG화학',
    '207940.KS': '삼성바이오로직스', '068270.KS': '셀트리온', '328150.KQ': '알테오젠', '178920.KQ': 'HLB', '000100.KS': '유한양행', '128940.KS': '한미약품', '009290.KS': '광동제약', '008930.KS': '한미사이언스',
    '105560.KS': 'KB금융', '055550.KS': '신한지주', '323410.KS': '카카오뱅크', '000810.KS': '삼성화재', '034730.KS': 'SK', '006800.KS': '미래에셋증권', '016360.KS': '삼성증권', '039490.KS': '키움증권',
    '005380.KS': '현대차', '000270.KS': '기아', '012330.KS': '현대모비스', '018880.KS': '한온시스템', '011790.KS': 'SKC', '010130.KS': '고려아연', '004020.KS': '현대제철', '017940.KS': 'E1',
    '096770.KS': 'SK이노베이션', '086520.KQ': '에코프로', '000150.KS': '두산', '004990.KS': '롯데케미칼',
    '352820.KS': '하이브', '035900.KS': 'JYP Ent.', '041510.KQ': '에스엠', '035760.KQ': 'CJ ENM', '140410.KQ': '와이지엔터테인먼트', '131970.KQ': '스튜디오드래곤', '237880.KQ': '클래시스', '187790.KQ': '테크윙',
    '251270.KS': '넷마블', '036570.KS': '크래프톤', '112040.KQ': '위메이드', '293490.KQ': '카카오게임즈', '078600.KQ': '펄어비스', '259960.KQ': '컴투스홀딩스', '052790.KQ': '아프리카TV', '042040.KQ': '케이사인',
    '^KS11': '코스피 지수',
    '005490.KS': 'POSCO홀딩스', '051900.KS': 'LG생활건강', '000720.KS': '현대건설',
}

top10_kospi_names = [
    '삼성전자', 'SK하이닉스', 'LG에너지솔루션', '삼성바이오로직스', '삼성전자우',
    '현대차', '기아', 'POSCO홀딩스', 'NAVER', 'LG화학'
]

# 이름을 키로, 티커를 값으로 하는 역방향 맵
name_ticker_map = {name: ticker for ticker, name in ticker_name_map.items()}
kospi_ticker = '^KS11'

# 4. 분석 로직 함수

@st.cache_data
def get_stock_data(tickers, period="3y"):
    #선택한 종목들의 과거 주가 데이터를 yfinance로 한 번에 가져옴.
    try:
        raw_data = yf.download(tickers, period=period, progress=False)
        if raw_data.empty: return None
        
        # yfinance 데이터 구조 처리 (Close, Adj Close 우선순위 선택)
        if isinstance(raw_data.columns, pd.MultiIndex):
            try:
                close_data = raw_data.loc[:, pd.IndexSlice['Close', :]]
                close_data.columns = close_data.columns.get_level_values(1)
            except KeyError:
                close_cols = [col for col in raw_data.columns if 'Close' in col[0]]
                if not close_cols:
                    adj_close_cols = [col for col in raw_data.columns if 'Adj Close' in col[0]]
                    if adj_close_cols:
                        close_data = raw_data[adj_close_cols]
                        close_data.columns = close_data.columns.get_level_values(1)
                    else:
                        return None
                else:
                    close_data = raw_data[close_cols]
                    close_data.columns = close_data.columns.get_level_values(1)
        elif 'Close' in raw_data.columns:
            close_data = raw_data[['Close']]
            if len(tickers) == 1: close_data.columns = [tickers[0]]
        elif 'Adj Close' in raw_data.columns:
            close_data = raw_data[['Adj Close']]
            if len(tickers) == 1: close_data.columns = [tickers[0]]
        else:
            return None
        
        close_data = close_data.dropna(how='all')
        return close_data if not close_data.empty else None
    except Exception:
        return None

@st.cache_data
def check_anomalies(stock_data_series, recent_days=7, contamination=0.03):
    
    #Isolation Forest 알고리즘을 사용하여 이상치(비정상적인 급등락)를 탐지.

    if stock_data_series is None or stock_data_series.empty or stock_data_series.isnull().all():
        return False
    
    # 수익률 계산 (가격 변동폭)
    returns = stock_data_series.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if returns.empty or len(returns) < recent_days:
        return False
        
    # 모델 학습을 위한 데이터 형태 변환
    data_for_model = returns.values.reshape(-1, 1)
    
    try:
        # Isolation Forest 모델 학습
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(data_for_model)
        predictions = model.predict(data_for_model)
        
        # 최근 7일 내에 이상치(-1)가 있는지 확인
        recent_predictions = predictions[-recent_days:]
        if -1 in recent_predictions: return True
        else: return False
    except ValueError:
        return False

def analyze_stocks(stock_data, kospi_data_df):

    # 주가 데이터를 기반으로 수익률, 변동성, 상관계수 등을 계산하고 이상치 탐지 결과를 종합.

    if stock_data is None or kospi_data_df is None or stock_data.empty or kospi_data_df.empty: return None
    
    returns = stock_data.pct_change().dropna(how='all')
    kospi_returns = kospi_data_df.iloc[:, 0].pct_change().dropna()
    
    if returns.empty or kospi_returns.empty: return None
    analysis = {}
    
    # 코스피 데이터와 날짜 인덱스 맞추기
    common_index = returns.index.intersection(kospi_returns.index)
    if common_index.empty: return None
    returns = returns.loc[common_index]
    kospi_returns = kospi_returns.loc[common_index]

    for stock_ticker in returns.columns:
        if returns[stock_ticker].isnull().all(): continue
        stock_returns = returns[stock_ticker].dropna()
        if stock_returns.empty: continue
        
        # 연율화된 지표 계산 (1년 거래일수=252일)
        avg_return = stock_returns.mean() * 252 # 연평균 수익률
        volatility = stock_returns.std() * np.sqrt(252) # 연 변동성
        correlation = stock_returns.corr(kospi_returns) # 시장 상관계수
        max_daily_gain = stock_returns.max()
        max_daily_loss = stock_returns.min()
        
        # 이상치 탐지 수행
        original_price_series = stock_data[stock_ticker]
        has_anomaly = check_anomalies(original_price_series, recent_days=7)

        analysis[stock_ticker] = {
            '연평균 수익률': avg_return,
            '연 변동성': volatility,
            '코스피 상관계수': correlation,
            '일일 최대 상승률': max_daily_gain,
            '일일 최대 하락률': max_daily_loss,
            '🚨 최근 7일내 이상신호': has_anomaly
        }
    return analysis

@st.cache_data
def get_pykrx_name_map():
    # pykrx를 통해 전체 종목 리스트를 가져와서 검색용 맵 생성
    pykrx_name_map = {}
    for market in ["KOSPI", "KOSDAQ"]:
        suffix = ".KS" if market == "KOSPI" else ".KQ"
        tickers = pykrx.stock.get_market_ticker_list(market=market)
        for t in tickers:
            name = pykrx.stock.get_market_ticker_name(t)
            if name: pykrx_name_map[name] = f"{t}{suffix}"
    return pykrx_name_map

# 5. UI 페이지 구성 함수 (Step 1, 2, 3)

def page_step1_propensity():
    # Step 1: 투자 성향 설문조사 페이지
    st.title("🎯 Step 1. 투자 성향 분석")
    st.markdown("다음 5가지 질문에 답변해주세요.")
    st.write("")

    with st.container(border=True):
        # 설문 문항
        q1 = st.radio("1. 당신의 연령은 어떻게 됩니까?", ['19세이하', '20 ~ 40세', '41 ~ 50세', '51 ~ 60세', '60세이상'], key='q1')
        st.write("")
        q2 = st.radio("2. 투자하고자 하는 자금의 투자 가능 기간은 얼마나 됩니까?", ['6개월 이내', '6개월 ~ 1년 이내', '1년 ~ 2년 이내', '2년 ~ 3년 이내', '3년 이상'], key='q2')
        st.write("")
        q3 = st.radio("3. 현재 투자하고자 하는 자금은 전체 금융자산 중 어느 정도의 비중을 차지합니까?", ['10% 이내', '10% ~ 20% 이내', '20% ~ 30% 이내', '30% ~ 40% 이내', '40% 초과'], key='q3')
        st.write("")
        q4 = st.radio("4. 다음 중 당신의 수입원을 가장 잘 나타내고 있는 것은 어느 것입니까?", [
                '현재 일정한 수입이 발생하고 있으며, 향후 현재 수준을 유지하거나 증가할 것으로 예상',
                '현재 일정한 수입이 발생하고 있으나, 향후 감소하거나 불안정할 것으로 예상',
                '현재 일정한 수입원이 없으며, 연금이 주 수입원임'
            ], key='q4')
        st.write("")
        q5 = st.radio("5. 만약 투자원금에 손실이 발생할 경우 다음 중 감수할 수 있는 손실 수준은 어느 것입니까?", [
                '무슨 일이 있어도 투자원금은 보전되어야 한다.',
                '10% 미만까지는 손실을 감수 할 수 있을 것 같다.',
                '20% 미만까지는 손실을 감수 할 수 있을 것 같다.',
                '기대수익이 높다면 위험이 높아도 상관하지 않겠다.'
            ], key='q5')

    # 점수 계산 로직 (각 답변에 가중치 부여)
    score = 0
    score += {'19세이하': 5, '20 ~ 40세': 5, '41 ~ 50세': 4, '51 ~ 60세': 3, '60세이상': 2}[q1]
    score += {'6개월 이내': 1, '6개월 ~ 1년 이내': 2, '1년 ~ 2년 이내': 3, '2년 ~ 3년 이내': 4, '3년 이상': 5}[q2]
    score += {'10% 이내': 1, '10% ~ 20% 이내': 2, '20% ~ 30% 이내': 3, '30% ~ 40% 이내': 4, '40% 초과': 5}[q3]
    score += {'현재 일정한 수입이 발생하고 있으며, 향후 현재 수준을 유지하거나 증가할 것으로 예상': 5, 
              '현재 일정한 수입이 발생하고 있으나, 향후 감소하거나 불안정할 것으로 예상': 3, 
              '현재 일정한 수입원이 없으며, 연금이 주 수입원임': 1}[q4]
    score += {'무슨 일이 있어도 투자원금은 보전되어야 한다.': 0, 
              '10% 미만까지는 손실을 감수 할 수 있을 것 같다.': 4, 
              '20% 미만까지는 손실을 감수 할 수 있을 것 같다.': 8, 
              '기대수익이 높다면 위험이 높아도 상관하지 않겠다.': 12}[q5]

    # 분석 완료 버튼
    if st.button("성향 분석 완료 및 다음 단계 👉", type="primary", use_container_width=True):
        # 점수에 따른 성향 분류
        if score <= 12: propensity = '안정형'
        elif score <= 18: propensity = '안정추구형'
        elif score <= 24: propensity = '위험중립형'
        else: propensity = '적극투자형'
        
        # 세션에 결과 저장 및 다음 단계로 이동
        st.session_state.propensity = propensity
        st.session_state.step = 2
        st.rerun()


def page_step2_selection():
    #Step 2: 분석할 종목 선택 페이지

    st.title("🔍 Step 2. 관심 종목 선택")
    st.info(f"당신의 투자 성향은 **[{st.session_state.propensity}]** 입니다.")
    st.write("분석하고 싶은 종목을 선택해주세요.")

    with st.container(border=True):
        analysis_type = st.radio("어떤 방식으로 종목을 찾으시겠습니까?", 
                                 ('관심 분야별 보기', 'KOSPI 상위 10', '직접 검색'), 
                                 horizontal=True)
        
        selected_tickers = []
        
        # 1. 분야별 선택
        if analysis_type == '관심 분야별 보기':
            sector_list = list(sectors.keys())
            selected_sector = st.selectbox("관심 분야 선택", sector_list)
            selected_tickers = sectors.get(selected_sector, [])
            selected_names = [ticker_name_map.get(t, t) for t in selected_tickers]
            st.success(f"**선택된 분야:** {selected_sector} ({len(selected_names)}개 종목)")
            
        # 2. 코스피 Top 10 선택
        elif analysis_type == 'KOSPI 상위 10':
            selected_stocks_names = st.multiselect("분석할 종목 선택", top10_kospi_names, default=top10_kospi_names)
            if selected_stocks_names:
                selected_tickers = [name_ticker_map[name] for name in selected_stocks_names if name in name_ticker_map]
            
        # 3. 직접 검색 (전체 종목 로드 필요)
        elif analysis_type == '직접 검색':
            try:
                if 'pykrx_map' not in st.session_state:
                    with st.spinner("전체 종목 리스트를 불러오는 중입니다..."):
                        st.session_state.pykrx_map = get_pykrx_name_map()
                
                pykrx_map = st.session_state.pykrx_map
                available_names = sorted(list(pykrx_map.keys()))
                selected_names = st.multiselect("종목명 검색", available_names)
                
                if selected_names:
                    for name in selected_names:
                        yfinance_ticker = pykrx_map.get(name)
                        if yfinance_ticker:
                            selected_tickers.append(yfinance_ticker)
                            # 새로운 종목이면 맵에 추가
                            if yfinance_ticker not in ticker_name_map:
                                ticker_name_map[yfinance_ticker] = name
            except Exception as e:
                st.error(f"종목 로드 실패: {e}")

    # 하단 버튼 (이전/분석실행)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("👈 이전 단계", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("데이터 분석 실행 🚀", type="primary", use_container_width=True):
            if not selected_tickers:
                st.warning("분석할 종목을 최소 1개 이상 선택해주세요.")
            else:
                with st.spinner("데이터를 다운로드하고 분석 중입니다..."):
                    fundamental_df = get_fundamental_data()
                    st.session_state.fundamental_data = fundamental_df

                    # 가격 데이터 다운로드 (선택종목 + 코스피지수)
                    all_tickers_to_download = list(set(selected_tickers + [kospi_ticker]))
                    stock_data_full = get_stock_data(all_tickers_to_download)

                    if stock_data_full is not None and not stock_data_full.empty:
                        if kospi_ticker in stock_data_full.columns:
                            # 코스피 데이터 분리
                            kospi_data = stock_data_full[[kospi_ticker]]
                            stock_data_prices = stock_data_full.drop(columns=[kospi_ticker], errors='ignore')

                            if not stock_data_prices.empty:
                                # 분석 실행
                                analysis_result = analyze_stocks(stock_data_prices, kospi_data)
                                st.session_state.analysis_result = analysis_result
                                st.session_state.stock_data_prices = stock_data_prices
                                st.session_state.step = 3
                                st.rerun()
                            else:
                                st.error("선택한 종목의 가격 데이터가 없습니다.")
                        else:
                            st.error("지수 데이터 로드 실패.")
                    else:
                        st.error("데이터 다운로드 실패.")


def page_step3_result():
    #Step 3: 분석 결과 출력 페이지

    propensity = st.session_state.propensity
    analysis = st.session_state.analysis_result
    stock_data_prices = st.session_state.stock_data_prices
    fundamental_data = st.session_state.fundamental_data

    st.title("📊 Step 3. 분석 결과 리포트")
    
    # 성향별 가이드 텍스트 및 정렬 기준 설정
    guide_text = ""
    if propensity == '안정형':
        guide_text = """
        **📋 안정형 투자자 가이드:**\n
        귀하는 원금 보존을 최우선으로 하는 투자자입니다. 
        단기적인 시세 차익보다는 예금 금리 수준 + α 의 안정적인 수익을 추구합니다.\n
        **추천 전략:** 
        - 변동성이 매우 낮고 재무구조가 탄탄한 대형 우량주 위주
        - 배당 수익률이 높은 종목(DIV)에 주목하세요.
        """
        sort_col = '연 변동성'
        ascending_sort = True
        
    elif propensity == '안정추구형':
        guide_text = """
        **📋 안정추구형 투자자 가이드:**\n
        귀하는 원금의 손실은 최소화하면서도, 예금보다는 높은 수익을 기대합니다.
        약간의 변동성은 감내할 수 있지만 큰 하락은 피하고 싶어합니다.\n
        **추천 전략:** 
        - 실적이 꾸준히 상승하는 우량주 분산 투자
        - PBR이 낮아 저평가된 종목을 찾아보세요.
        """
        sort_col = '연 변동성' 
        ascending_sort = True
        
    elif propensity == '위험중립형':
        guide_text = """
        **📋 위험중립형 투자자 가이드:**\n
        귀하는 수익을 위해 어느 정도의 위험(원금 손실 가능성)을 충분히 감수할 수 있습니다.
        시장 평균 이상의 수익률을 목표로 하며, 주식 시장의 등락을 이해합니다.\n
        **추천 전략:** 
        - 성장 가능성이 높은 섹터(IT, 반도체 등) 주도주 투자
        - 연평균 수익률과 PER 지표를 균형 있게 고려하세요.
        """
        sort_col = '연평균 수익률'
        ascending_sort = False
        
    else: # 적극투자형
        guide_text = """
        **📋 적극투자형 투자자 가이드:**\n
        귀하는 높은 수익을 위해서라면 원금의 상당 부분 손실 위험도 감수하는 공격적인 투자자입니다.
        단기적인 변동성을 기회로 활용하며 적극적으로 매매에 임합니다.\n
        **추천 전략:** 
        - 높은 변동성을 가진 성장주, 테마주 트레이딩
        - 최근 거래량이 급증하거나 이상 신호가 감지된 종목도 적극 검토 (단, 리스크 관리 필수)
        """
        sort_col = '연평균 수익률'
        ascending_sort = False

    st.info(guide_text)
    
    if not analysis:
        st.error("분석 결과가 없습니다.")
        return

    # 1. 3년 수익률 비교 차트 (정규화)
    st.subheader("📈 3년간 주가 수익률 추이 (비교)")
    if stock_data_prices is not None:
        price_df_to_plot = stock_data_prices.copy()
        price_df_to_plot.columns = price_df_to_plot.columns.map(ticker_name_map.get)
        try:
            # 시작점을 100으로 맞추어 비교 (정규화)
            normalized_df = (price_df_to_plot.fillna(method='bfill') / price_df_to_plot.fillna(method='bfill').iloc[0]) * 100
            st.line_chart(normalized_df, use_container_width=True)
        except Exception:
            st.warning("차트를 그릴 수 없습니다.")
    
    st.markdown("---")

    # 2. 분석 결과 데이터프레임 생성 및 병합
    df = pd.DataFrame.from_dict(analysis, orient='index')
    
    # 재무 정보(PER, PBR 등) 병합
    if fundamental_data is not None and not fundamental_data.empty:
        df['ticker_clean'] = df.index.map(lambda x: str(x).split('.')[0])
        try:
            df = df.reset_index()
            # Left Join 수행
            df = pd.merge(df, fundamental_data, left_on='ticker_clean', right_index=True, how='left')
            df = df.set_index('index')
            df = df.drop(columns=['ticker_clean'], errors='ignore')
        except Exception as e:
            st.warning(f"재무 정보 병합 실패: {e}")

    # 종목명 및 링크 생성
    df['종목명'] = df.index.map(lambda x: ticker_name_map.get(x, x))
    df['상세정보'] = df.index.map(lambda x: f"https://finance.naver.com/item/main.naver?code={x.split('.')[0]}")
    df = df.set_index('종목명')

    # 성향에 맞게 데이터 정렬
    df_sorted = df.sort_values(by=sort_col, ascending=ascending_sort)

    anomaly_stocks_df = df[df['🚨 최근 7일내 이상신호'] == True]
    
    # 1순위 추천 종목 표시
    best_stock = df_sorted.index[0] if not df_sorted.empty else "없음"
    st.success(f"🏆 1순위 추천 종목: **{best_stock}**")

    # 테이블 컬럼 포맷 설정
    column_config = {
        "상세정보": st.column_config.LinkColumn("자세히 보기", display_text="바로가기"),
        "PER": st.column_config.NumberColumn("PER(주가수익률)", format="%.2f"),
        "PBR": st.column_config.NumberColumn("PBR(주가순자산비율)", format="%.2f"),
        "DIV": st.column_config.NumberColumn("배당수익률", format="%.2f%%"),
    }
    base_formatter = {
        '연평균 수익률': '{:.2%}', '연 변동성': '{:.2%}', '코스피 상관계수': '{:.2f}',
        '일일 최대 상승률': '{:.2%}', '일일 최대 하락률': '{:.2%}',
    }
    
    # 포트폴리오 테이블 출력
    st.subheader(f"📄 맞춤 포트폴리오 ({propensity})")
    
    # 표시용 데이터프레임 복사 및 텍스트 치환
    df_display = df_sorted.copy()
    df_display['🚨 최근 7일내 이상신호'] = df_display['🚨 최근 7일내 이상신호'].apply(lambda x: "🔥 감지됨" if x else "✅ 정상")
    
    st.dataframe(df_display.style.format(base_formatter), column_config=column_config, use_container_width=True)

    # 이상치 감지 종목 별도 표시
    if not anomaly_stocks_df.empty:
        st.subheader("🚨 투자 유의 필요 (이상 급등락 감지)")
        st.write("최근 7일 이내에 통계적으로 비정상적인 주가 흐름이 감지된 종목입니다.")
        anomaly_display = anomaly_stocks_df.copy()
        anomaly_display['🚨 최근 7일내 이상신호'] = "🔥 감지됨"
        st.dataframe(
            anomaly_display.sort_values(by='연 변동성', ascending=False).style.format(base_formatter), 
            column_config=column_config, 
            use_container_width=True
        )

    st.markdown("---")
    
    # 차트 섹션
    st.subheader("🔎 상세 차트 ")
    ticker_list = list(analysis.keys())
    name_list = [ticker_name_map.get(t, t) for t in ticker_list]
    selected_name = st.selectbox("차트를 확인할 종목 선택:", name_list)
    
    if selected_name:
        # 선택된 이름으로 티커 찾기
        selected_ticker = [k for k, v in ticker_name_map.items() if v == selected_name][0]
        with st.spinner("차트 그리는 중..."):
            fig = plot_candlestick(selected_ticker, selected_name)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("차트 데이터를 불러올 수 없습니다.")

    st.write("")
    with st.container(border=True):
        st.markdown("#### 💡 투자 전 필독사항")
        st.markdown("""<small>본 서비스는 과거 데이터를 기반으로 합니다. 모든 투자의 책임은 투자자 본인에게 있습니다.</small>""", unsafe_allow_html=True)

    if st.button("🔄 처음부터 다시 하기", use_container_width=True):
        st.session_state.step = 1
        st.session_state.selected_tickers = []
        st.rerun()

# 6. 메인 실행 컨트롤러

def main():
    # 현재 단계(step)에 따라 해당 페이지 함수를 호출
    if st.session_state.step == 1:
        page_step1_propensity()
    elif st.session_state.step == 2:
        page_step2_selection()
    elif st.session_state.step == 3:
        page_step3_result()

if __name__ == "__main__":
    main()