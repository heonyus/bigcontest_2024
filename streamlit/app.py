import streamlit as st

st.set_page_config(page_title="🍊 제주 맛집 추천 챗봇", page_icon="🍊", layout="wide")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import pandas as pd
import torch
import faiss
import requests
import folium
from streamlit_folium import st_folium
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
from tqdm import tqdm
import logging
import sys
import time as time_module
from datetime import datetime


# 로깅 레벨 설정을 DEBUG로 변경
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # stdout으로 변경
        logging.FileHandler("app_log.log", encoding='utf-8')  # 인코딩 지정
    ]
)

# 1. 기본 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
KAKAO_API_KEY = st.secrets["KAKAO_API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# transformers import 수정
try:
    from transformers import AutoTokenizer, AutoModel
    logging.info("Transformers 라이브러리 로드 성공")
except ImportError as e:
    logging.error(f"Transformers 라이브러리 로드 실패: {e}")
    st.error("필요한 라이브러리 로드에 실패했. 관리자에게 문의하세요.")
    st.stop()

# 모델 로드 부분 수정
try:
    embedding_model_name = "jhgan/ko-sroberta-multitask"
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    embedding_model = AutoModel.from_pretrained(embedding_model_name)
    
    if torch.cuda.is_available():
        embedding_model = embedding_model.to(device)
        logging.info("모델을 GPU로 로드했습니다.")
    else:
        logging.info("모델을 CPU로 로드했습니다.")
except Exception as e:
    logging.error(f"모델 로드 실패: {e}")
    st.error("모델 로드에 실패했습니다. 관리자에게 문의하세요.")
    st.stop()

embedding_data_path = './data/JEJU_with_embeddings_final.csv'

# 임베딩 문자열을 numpy 배열로 변환하는 함수
def str_to_array(embedding_str):
    try:
        # 문자열에서 숫자만 추출
        numbers = [float(x) for x in embedding_str.strip('[]').split()]
        return np.array(numbers)
    except Exception as e:
        logging.error(f"임베딩 변환 오류: {e}")
        return np.zeros(768)  # 기본 임베 차

# 2. FAISS 인덱스 생성 함수 (CPU 전용)
def create_faiss_index(embeddings):
    try:
        dimension = embeddings.shape[1]
        logging.info(f"FAISS 인덱스 생성 시작 (차원: {dimension})")
        
        # CPU 버전 인덱스만 사용
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        logging.info("FAISS CPU 인덱스 생성 완료")
        return index
    except Exception as e:
        logging.error(f"FAISS 인덱스 생성 실패: {e}")
        raise

# 앱 시작 부분에 추가
@st.cache_resource
def load_models():
    """모델 로드를 캐시하여 재사용"""
    try:
        embedding_model_name = "jhgan/ko-sroberta-multitask"
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        embedding_model = AutoModel.from_pretrained(embedding_model_name)
        
        if torch.cuda.is_available():
            embedding_model = embedding_model.to(device)
            logging.info("모델을 GPU로 로드했습니다.")
        else:
            logging.info("모델을 CPU로 로드했습니다.")
            
        return tokenizer, embedding_model
    except Exception as e:
        logging.error(f"모델 로드 실패: {e}")
        st.error("모델 로드에 실패했습니다. 관리자에게 문의하세요.")
        st.stop()

@st.cache_data
def load_and_process_data():
    """데이터 로드 및 전처리를 캐시하여 재사용"""
    df = pd.read_csv(embedding_data_path, encoding='utf-8-sig')
    
    # 장소와 카테고리 임베딩을 별도의 FAISS 인덱스로 생성
    location_embeddings = np.stack(df['장소_임베딩'].apply(str_to_array).to_numpy()).astype(np.float32)
    category_embeddings = np.stack(df['카테고리_임베딩'].apply(str_to_array).to_numpy()).astype(np.float32)
    
    # L2 정규화 적용
    faiss.normalize_L2(location_embeddings)
    faiss.normalize_L2(category_embeddings)
    
    # 각각의 FAISS 인덱스 생성
    index_location = faiss.IndexFlatIP(location_embeddings.shape[1])
    index_category = faiss.IndexFlatIP(category_embeddings.shape[1])
    
    index_location.add(location_embeddings)
    index_category.add(category_embeddings)
    
    return df, index_location, index_category

# 메인 코드에서 사용
tokenizer, embedding_model = load_models()
df, index_location, index_category = load_and_process_data()

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()[0]

def search_with_faiss(location, category, top_k=5):
    logging.info(f"=== FAISS 검색 시작 ===")
    logging.info(f"검색 조건 - 위치: {location}, 카테고리: {category}, top_k: {top_k}")
    
    try:
        # 1. 카테고리 검색 (더 많은 후보군 검색)
        category_emb = embed_text(category)
        category_query = np.expand_dims(category_emb, axis=0).astype(np.float32)
        faiss.normalize_L2(category_query)
        category_distances, category_indices = index_category.search(category_query, top_k * 4)
        
        # 카테고리 유사도가 높은 항목만 필터링 (임계값: 0.7)
        category_threshold = 0.7
        valid_category_mask = (1 - category_distances[0]) >= category_threshold
        category_candidates = category_indices[0][valid_category_mask]
        
        if len(category_candidates) == 0:
            logging.warning(f"카테고리 '{category}'와 일치하는 결과가 없습니다.")
            return None, None
            
        logging.info(f"카테고리 검색 결과 수: {len(category_candidates)}")
        
        # 2. 위치 검색
        location_emb = embed_text(location)
        location_query = np.expand_dims(location_emb, axis=0).astype(np.float32)
        faiss.normalize_L2(location_query)
        location_distances, location_indices = index_location.search(location_query, top_k * 4)
        
        # 위치 유사도가 높은 항목만 필터링 (임계값: 0.7)
        location_threshold = 0.7
        valid_location_mask = (1 - location_distances[0]) >= location_threshold
        location_candidates = location_indices[0][valid_location_mask]
        
        if len(location_candidates) == 0:
            logging.warning(f"위치 '{location}'와 일치하는 결과가 없습니다.")
            return None, None
            
        logging.info(f"위치 검색 결과 수: {len(location_candidates)}")
        
        # 3. 교집합 찾기
        common_indices = np.intersect1d(category_candidates, location_candidates)
        
        if len(common_indices) == 0:
            logging.warning("위치와 카테고리 조건을 모두 만족하는 결과가 없습니다.")
            return None, None
            
        logging.info(f"교집합 결과 수: {len(common_indices)}")
        
        # 4. 최종 결과 정렬 (카테고리 유사도 기준)
        final_results = []
        for idx in common_indices:
            cat_sim = 1 - category_distances[0][np.where(category_indices[0] == idx)[0][0]]
            loc_sim = 1 - location_distances[0][np.where(location_indices[0] == idx)[0][0]]
            final_results.append((idx, cat_sim, loc_sim))
        
        # 카테고리 유사도로 정렬
        final_results.sort(key=lambda x: x[1], reverse=True)
        final_results = final_results[:top_k]
        
        # 결과 로깅
        logging.info(f"최종 검색 결과:")
        for idx, cat_sim, loc_sim in final_results:
            logging.info(f"- {df.iloc[idx]['가맹점명']}")
            logging.info(f"  카테고리: {df.iloc[idx]['Categories']}")
            logging.info(f"  카테고리 유사도: {cat_sim:.4f}")
            logging.info(f"  위치 유사도: {loc_sim:.4f}")
        
        final_indices = [r[0] for r in final_results]
        final_scores = [r[1] for r in final_results]  # 카테고리 유사도를 최종 점수로 사용
        
        return np.array(final_scores), np.array(final_indices)
        
    except Exception as e:
        logging.error(f"FAISS 검색 중 오류 발생: {str(e)}", exc_info=True)
        raise

def get_coords_from_address(address):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": address}
    try:
        logging.info(f"카카오 API 요청 - 주소: {address}")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        result = response.json()
        logging.info(f"카카오 API 응답: {result}")
        
        if result['documents']:
            x = float(result['documents'][0]['x'])
            y = float(result['documents'][0]['y'])
            coords = {'lat': y, 'lng': x}
            logging.info(f"좌표 변환 성공: {coords}")
            return coords
    except Exception as e:
        logging.error(f"[ERROR] 좌표 변환 실패: {e}", exc_info=True)
    return None

def get_coordinates(address):
    """카카오 API를 사용하여 주소의 좌표를 가져옵니다."""
    try:
        url = 'https://dapi.kakao.com/v2/local/search/address.json'
        headers = {'Authorization': f'KakaoAK {KAKAO_API_KEY}'}
        params = {'query': address}
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            result = response.json()
            if result['documents']:
                x = float(result['documents'][0]['x'])  # 경도
                y = float(result['documents'][0]['y'])  # 위도
                return y, x
        return None, None
    except Exception as e:
        logging.error(f"좌표 변환 중 오류 발생: {str(e)}")
        return None, None

def display_map(df):
    try:
        # 상위 3개 결과만 사용하도록 수정
        df = df.head(3)
        logging.info(f"지도 생성 시작 - 최종 추천 식당 수: {len(df)}")
        
        # 주소를 좌표로 변환
        valid_data = []
        for idx, row in df.iterrows():
            coords = get_coords_from_address(row['주소'])
            if coords:
                # 유사도 점수 포맷팅을 위한 처리
                similarity_score = row.get('similarity_score', 'N/A')
                if isinstance(similarity_score, float):
                    similarity_score = f"{similarity_score:.4f}"
                
                valid_data.append({
                    'lat': coords['lat'],
                    'lng': coords['lng'],
                    'name': row['가맹점명'],
                    'address': row['주소'],
                    'category': row['Categories'],
                    'similarity_score': similarity_score,
                    'local_usage': row['현지인이용건수비중']
                })
                logging.info(f"식당 {idx+1} 좌표 변환 성공: {row['가맹점명']} - {coords}")
        
        if not valid_data:
            logging.error("추천된 식당의 유효한 좌표가 없습니다.")
            return None
            
        # 중심점 계산
        center_lat = sum(data['lat'] for data in valid_data) / len(valid_data)
        center_lng = sum(data['lng'] for data in valid_data) / len(valid_data)
        
        # 지도 생성
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=13,
            width='100%',
            height='100%'
        )
        
        # 마커 추가 - 상세 정보 포함
        for data in valid_data:
            popup_content = f"""
            <div style='width: 200px'>
                <h4>{data['name']}</h4>
                <p>주소: {data['address']}</p>
                <p>업종: {data['category']}</p>
                <p>유사도: {data['similarity_score']}</p>
                <p>현지인 이용: {data['local_usage']:.1f}%</p>
            </div>
            """
            
            folium.Marker(
                location=[data['lat'], data['lng']],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=data['name'],
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        
        logging.info(f"지도 생성 완료 - 표시된 식당 수: {len(valid_data)}")
        return m
        
    except Exception as e:
        logging.error(f"지도 생성 중 오류 발생: {str(e)}", exc_info=True)
        return None

# 3. 응답 생성 함수 (추천 메시 생성)
def generate_response(prompt, filtered_df, time, local_choice):
    try:
        logging.info(f"\n{'='*50}\n응답 생성 시작\n{'='*50}")
        
        # 상위 3개 식당 정보 준비
        top_3_restaurants = filtered_df.head(3)
        restaurant_details = []
        
        for _, row in top_3_restaurants.iterrows():
            usage_level = row['이용건수구간']
            local_usage = row['현지인이용건수비중']
            time_info = ""
            if time:
                time_usage = row[f'{time}이용건수비중']
                time_info = f", {time} 이용 비중: {time_usage:.1f}%"
            
            detail = {
                '이름': row['가맹점명'],
                '주소': row['주소'],
                '업종': row['Categories'],
                '이용건수구간': usage_level,
                '현지인이용비중': f"{local_usage:.1f}%",
                '시간대정보': time_info
            }
            restaurant_details.append(detail)
        
        # 프롬프트 구성
        full_prompt = f"""
사용자 질문: {prompt}

추천 식당 정보:
{'-'*50}
"""
        for i, detail in enumerate(restaurant_details, 1):
            full_prompt += f"""
{i}. {detail['이름']}
- 주소: {detail['주소']}
- 업종: {detail['업종']}
- 이용건수: {detail['이용건수구간']} (5_75~90%는 상위 10~25%, 6_90~100%는 상위 10% 구간)
- 현지인 이용 비중: {detail['현지인이용비중']}{detail['시간대정보']}
"""

        full_prompt += f"""
{'-'*50}

위 정보를 바탕으로 사용자의 질문에 답변해주세요. 
- 이용건수 구간을 명확히 설명해주세요
- 현지인 이용 비중도 언급해주세요
- 각 식당의 특징을 자연스럽게 설명해주세요
- 친근하고 자연스러운 어투로 답변해주세요
"""

        logging.info(f"\nLLM 입력 프롬프트:\n{full_prompt}")
        response = model.generate_content(full_prompt)
        logging.info(f"\nLLM 생성 응답:\n{response.text}")
        
        return response.text
        
    except Exception as e:
        logging.error(f"응답 생성 중 오류 발생: {str(e)}", exc_info=True)
        return "추천을 생성하는 중 오류가 발생했습니다."

def filter_data_by_criteria(df, location, category, time_periods=None, conditions=None):
    try:
        logging.info(f"\n{'='*50}\n필터링 프로세스 시작\n{'='*50}")
        logging.info(f"입력 파라미터:\n- 위치: {location}\n- 카테고리: {category}\n- 시간대: {time_periods}\n- 조건: {conditions}")
        logging.info(f"초기 데이터 크기: {len(df)} 행")
        
        filtered_df = df.copy()
        
        # 1. FAISS 임베딩 기반 검색
        logging.info("1. FAISS 임베딩 기반 검색 시작")
        
        # 문자열을 numpy 배열로 변환하는 함수
        def parse_embedding(embedding_str):
            # 문자열에서 숫자만 추출
            numbers = [float(num) for num in embedding_str.strip('[]').split()]
            return np.array(numbers)

        try:
            # CSV 파일 읽기 시 인코딩 설정
            if '장소_임베딩' not in filtered_df.columns:
                embedding_column = [col for col in filtered_df.columns if '임베딩' in col][0]
                filtered_df = filtered_df.rename(columns={embedding_column: '장소_임베딩'})
            
            # 데이터프레임의 임베딩을 numpy 배열로 변환
            embeddings = []
            for emb_str in filtered_df['장소_임베딩'].values:
                try:
                    numbers = [float(num) for num in emb_str.strip('[]').split()]
                    embeddings.append(numbers)
                except Exception as e:
                    logging.error(f"개별 임베딩 변환 오류: {e}")
                    embeddings.append(np.zeros(768))
            
            # numpy 배열로 변환하고 float32 타입으로 명시적 변환
            embeddings = np.array(embeddings, dtype=np.float32)
            logging.info(f"임베딩 배열 shape: {embeddings.shape}")

            # FAISS 인덱스 생성 및 벡터 추가
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            
            # L2 정규화 전에 복사본 생성
            embeddings_normalized = embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            
            # 정규화된 벡터 추가
            index.add(embeddings_normalized)

            # 쿼리 임베딩 준비
            location_mask = filtered_df['파싱된_장소'].str.contains(location, na=False)
            if location_mask.any():
                location_emb = parse_embedding(filtered_df[location_mask].iloc[0]['장소_임베딩'])
                query_emb = location_emb.reshape(1, -1).astype(np.float32)
                
                # 쿼리 벡터 정규화
                query_normalized = query_emb.copy()
                faiss.normalize_L2(query_normalized)

                # 유사도 검색
                k = min(1000, len(filtered_df))
                similarities, indices = index.search(query_normalized, k)

                logging.info(f"FAISS 검색 결과 - 최대 유사도: {similarities[0][0]:.4f}, 최소 유사도: {similarities[0][-1]:.4f}")

                # 유사도 임계값 적용
                threshold = 0.3
                valid_indices = indices[0][similarities[0] > threshold]
                filtered_df = filtered_df.iloc[valid_indices].copy()
                filtered_df['similarity_score'] = similarities[0][similarities[0] > threshold]

                logging.info(f"임베딩 기반 필터링 후 데이터 수: {len(filtered_df)}")
            else:
                logging.warning(f"위치 '{location}'에 해당하는 데이터를 찾을 수 없습니다.")
                return pd.DataFrame()

        except Exception as e:
            logging.error(f"임베딩 처리 중 오류 발생: {str(e)}")
            return pd.DataFrame()

        # 2. 시간대 필터링
        if time_periods:
            logging.info("2. 시간대 필터 시작")
            time_conditions = []
            for time in time_periods:
                if time == '아침':
                    time_conditions.append(filtered_df['5시11시이용건수비중'] > 0)
                elif time == '점심':
                    time_conditions.append(filtered_df['12시13시이용건수비중'] > 0)
                elif time == '저녁':
                    time_conditions.append(filtered_df['18시22시이용건수비중'] > 0)
                elif time == '밤':
                    time_conditions.append(filtered_df['23시4시이용건수비중'] > 0)
            
            if time_conditions:
                combined_condition = time_conditions[0]
                for condition in time_conditions[1:]:
                    combined_condition |= condition
                filtered_df = filtered_df[combined_condition]
                logging.info(f"시간대 필터링 후 데이터 수: {len(filtered_df)}")
        
        # 3. 조건 필터링
        if conditions:
            logging.info("3. 추가 조건 필터링 시작")
            for condition in conditions:
                before_count = len(filtered_df)
                
                if '이용건수 상위' in condition.lower():
                    if '10%' in condition:
                        filtered_df = filtered_df[filtered_df['이용건수구간'].isin(['5_75~90%', '6_90~100%'])]
                    elif '20%' in condition:
                        filtered_df = filtered_df[filtered_df['이용건수구간'].isin(['4_50~75%', '5_75~90%', '6_90~100%'])]
                    logging.info(f"이용건수 필링 - {condition} 적용 후: {len(filtered_df)}개")
                
                elif '현지인' in condition.lower():
                    filtered_df = filtered_df.sort_values('현지인이용건수비중', ascending=False)
                    logging.info(f"현지인 이용 비중 정렬 - 최대값: {filtered_df['현지인이용건수비중'].max():.2f}")
                
                elif '요일' in condition.lower():
                    day_columns = {
                        '월요일': '월요일이용건수비중',
                        '화요일': '화요일이용건수비중',
                        '수요일': '수요일이용건수비중',
                        '목요일': '목요일이용건수비중',
                        '금요일': '금요일이용건수비중',
                        '토요일': '토요일이용건수비중',
                        '일요일': '일요일이용건수비중'
                    }
                    
                    for day, column in day_columns.items():
                        if day in condition:
                            filtered_df = filtered_df.sort_values(column, ascending=False)
                            logging.info(f"{day} 이용 비중 - 최대값: {filtered_df[column].max():.2f}")
                
                elif '대' in condition:
                    age_columns = {
                        '20대': '최근12개월20대이하회원수비중',
                        '30대': '최근12개월30대회원수비중',
                        '40대': '최근12개월40대회원수비중',
                        '50대': '최근12개월50대회원수비중',
                        '60대': '최근12개월60대이상회원수비중'
                    }
                    
                    for age, column in age_columns.items():
                        if age in condition:
                            if '상' in condition:
                                threshold = float(condition.split('%')[0].split('이상')[0].strip())
                                filtered_df = filtered_df[filtered_df[column] >= threshold]
                            logging.info(f"{age} 필터링 - {condition} 적용 후: {len(filtered_df)}개")
                
                logging.info(f"조건 '{condition}' 적용 후 데이터 수 변화: {before_count} -> {len(filtered_df)}")
        
        # 각 필터링 단계마다 상세 로깅 추가
        if time_periods:
            for time in time_periods:
                before_count = len(filtered_df)
                # ... 기존 시간대 필터링 로직 ...
                after_count = len(filtered_df)
                logging.info(f"시간대 '{time}' 필터링 결과: {before_count} -> {after_count} 행")
        
        # 조건 필터링에 상세 로깅 추가
        if conditions:
            for condition in conditions:
                before_count = len(filtered_df)
                # ... 기존 조건 필터링 로직 ...
                after_count = len(filtered_df)
                logging.info(f"조건 '{condition}' 필터링 결과: {before_count} -> {after_count} 행")
                
                # 조건별 상세 정보 로깅
                if '이용건수 상위' in condition.lower():
                    logging.info(f"이용건수 구간 분포:\n{filtered_df['이용건수구간'].value_counts()}")
                elif '현지인' in condition.lower():
                    logging.info(f"현지 이용 비중 통계:\n- 평균: {filtered_df['현지인이용건수비중'].mean():.2f}\n- 최대: {filtered_df['현지인이용건수비중'].max():.2f}")
        
        # 최종 결과에 대한 상세 정보 로깅
        logging.info(f"\n{'='*50}\n최종 결과 상세 정보\n{'='*50}")
        for idx, row in filtered_df.head().iterrows():
            logging.info(f"\n식당 {idx+1} 상세정보:")
            logging.info(f"- 이름: {row['가맹점명']}")
            logging.info(f"- 주소: {row['주소']}")
            logging.info(f"- 카테고리: {row['Categories']}")
            logging.info(f"- 유사도 점수: {row.get('similarity_score', 'N/A'):.4f}")
            logging.info(f"- 현지인 이용 비중: {row['현지인이용건수비중']:.2f}%")
            if time_periods:
                for time in time_periods:
                    if time == '아침':
                        logging.info(f"- 아침 이용 비중: {row['5시11시이용건수비중']:.2f}%")
                    elif time == '점심':
                        logging.info(f"- 점심 이용 비중: {row['12시13시이용건수비중']:.2f}%")
                    elif time == '저녁':
                        logging.info(f"- 저녁 이용 비중: {row['18시22시이용건수비중']:.2f}%")
                    elif time == '밤':
                        logging.info(f"- 심야 이용 비중: {row['23시4시이용건수비중']:.2f}%")
        
        return filtered_df

    except Exception as e:
        logging.error(f"필터링 중 오류 발생: {str(e)}", exc_info=True)
        return pd.DataFrame()

# 텍스트 임베딩 생성 함수
def get_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        
        # [CLS] 토큰의 임베딩 사용
        embedding = outputs.last_hidden_state[0][0].cpu().numpy()
        return embedding
        
    except Exception as e:
        logging.error(f"임베딩 생성 중 오류 발생: {str(e)}")
        return np.zeros(768)  # 기본 임베딩 차원

def parse_user_input_to_json(user_input, retries=3):
    logging.info(f"사용자 입력 파싱 시작: {user_input}")
    
    prompt = f"""
    제주도 맛집 데이터를 기반으로 사용자 입력을 구조화된 JSON으로 변환하는 작업입니다.

    다음 단계에 따라 분석해주세요:

    1. 장소 정보 추출
    - 제주시/서귀포시 구분
    - 읍/면/동 단위까지 포함
    - 예: "제주시 노형동", "서귀포시 색달동"

    2. 업종 분류
    - 주요 카테고리: "육류,고기요리", "해물,생선요리", "가정식", "단품요리 전문" 등
    - 구체적 업종으로 매핑

    3. 시간대 분석
    - 아침(5-11시), 점심(12-13시), 저녁(17-21시), 밤(22-23시)
    - 해당하는 시간대 배열로 표현

    4. 조건 분석 (아래 조건만 사용 가능)
    - 이용건수: "이용건수 상위 10%", "이용건수 상위 20%"
    - 연령대: "20대 이용 비중 30% 이상", "30대 이용 비중 30% 이상" 등
    - 요일: "월요일 이용 비중 가장 높음", "화요일 이용 비중 가장 높음" 등
    - 현지인: "현지인 이용 비중 가장 높음"

    ※ 부모님/가족 관련 키워드가 나오면 반드시 "50대 이용 비중 30% 이상"으로 변환

    출력 형식:
    {{
        "장소": "지역명",
        "Categories": "업종",
        "시간대": ["아침", "점심", "저녁", "밤"],
        "조건": ["조건1", "조건2", ...]
    }}

    예시 입력/출력:
    1. 입력: "제주시 노형동에 있는 단품요리 전문점 중 이용건수가 상위 10%에 속하고 현지인 이용 비중이 가장 높은 곳은?"
    → {{
        "장소": "제주시 노형동",
        "Categories": "단품요리 전문",
        "시간대": [],
        "조건": ["이용건수 상위 10%", "현지인 이용 비중 가장 높음"]
    }}

    2. 입력: "부모님과 함께 갈 서귀포시 색달동의 단품요리 전문점 추천해줘"
    → {{
        "장소": "서귀포시 색달동",
        "Categories": "단품요리 전문",
        "시간대": [],
        "조건": ["50대 이용 비중 30% 이상"]
    }}

    현재 입력: {user_input}
    위 예시들을 참고하여 JSON 형식으로만 출력해주세요.
    """
    
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            json_str = response.text.strip().replace("```json", "").replace("```", "").strip()
            logging.info(f"LLM 응답: {json_str}")
            
            parsed_input = eval(json_str)
            if "장소" in parsed_input and "Categories" in parsed_input:
                logging.info(f"파싱 성공: {parsed_input}")
                return parsed_input
        except Exception as e:
            logging.error(f"JSON 파싱 시도 {attempt + 1} 실패: {str(e)}")
            if attempt == retries - 1:
                raise
    return None

# 채팅 기록 초기화 함수
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "어떤 식당을 찾으시나요?"}]
    logging.info("채팅 기록 초기화")

# Streamlit UI 설정
st.title("🍊 제주 맛집 추천 챗봇")

# 세션 상태 초기화 부분
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "어떤 식당을 찾으시나요?"}]
    
if "map_data" not in st.session_state:
    st.session_state.map_data = None

# 사이드바에 초기화 버튼 추가
st.sidebar.button('🗑️ 대화 기록 지우기', on_click=clear_chat_history)

# 이전 메시지들 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력 처리 부분
if prompt := st.chat_input("예: 애월의 저녁 상위 10% 고기집 추천해줘"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    try:
        parsed_input = parse_user_input_to_json(prompt)
        if parsed_input:
            location = parsed_input["장소"]
            category = parsed_input["Categories"]
            time = parsed_input.get("시간대", [])
            conditions = parsed_input.get("조건", [])

            result_df = filter_data_by_criteria(df, location, category, time_periods=time, conditions=conditions)

            # 응답 생성
            with st.chat_message("assistant"):
                generated_response = generate_response(prompt, result_df, ", ".join(time), "맛집")
                st.write(generated_response)
                st.session_state.messages.append({"role": "assistant", "content": generated_response})

                # 지도 표시
                if result_df is not None and not result_df.empty:
                    with st.spinner('지도를 생성하는 중...'):
                        m = display_map(result_df)
                        if m:
                            st.write("## 추천된 식당 위치")
                            st_folium(m, width=700, height=500)
                            st.session_state.map_data = m  # 지도 데이터 저장
                        else:
                            st.warning("죄송합니다. 지도를 표시할 수 없습니다.")

        else:
            with st.chat_message("assistant"):
                error_message = "장소와 업종을 정확히 입력해주세요."
                st.warning(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    except Exception as e:
        with st.chat_message("assistant"):
            error_message = "처리 중 오류가 발생했습니다. 다시 시도해주세요."
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
        logging.error(f"처리 중 오류 발생: {str(e)}", exc_info=True)

# 저장된 지도 데이터가 있으면 표시
if st.session_state.map_data is not None:
    st.write("## 추천된 식당 위치")
    st_folium(st.session_state.map_data, width=700, height=500)




















