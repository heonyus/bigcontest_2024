import streamlit as st
# Streamlit 페이지 설정은 가장 먼저 호출되어야 합니다.
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
import time as time_module  # time 모듈에 별칭 부여
from datetime import datetime
import ast
import re
from geopy.distance import geodesic
import math  # 파일 상단의 다른 import문들과 함께 추가

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

embedding_data_path = './data/JEJU_with_embeddings_donghyeok.csv'

# 임베딩 문자열을 numpy 배열로 변환하는 함수
def str_to_array(embedding_str):
    """문자열을 numpy 배열로 변환하는 함수"""
    try:
        if isinstance(embedding_str, str):
            # 문자에서 숫자만 추출
            numbers = [float(num) for num in embedding_str.strip('[]').split()]
            return np.array(numbers)
        else:
            # 문자열이 아닌 경우, 예를 들어 NaN, float 등
            logging.error(f"임베딩 변환 오류: {embedding_str}는 문자열이 아닙니다.")
            return np.zeros(768)  # 기본 크기의 0 배열 반환
    except Exception as e:
        logging.error(f"임베딩 변환 오류: {e}")
        return np.zeros(768)  # 기본 크기의 0 배열 반환

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
        st.error("모델 로드 실패습니다. 관리자에게 문의하세요.")
        st.stop()

# 전역 변수로 키워드 임베딩 저장
keyword_embeddings_dict = {}
initialized = False

def embed_text(text):
    """텍스트를 임베딩 벡터로 변환"""
    try:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = embedding_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        return embedding
    except Exception as e:
        logging.error(f"텍스트 임베딩 생성 오류: {e}")
        return np.zeros(768)

@st.cache_resource
def load_and_process_data():
    """데이터 로드 및 전처리를 캐시하여 재사용"""
    try:
        # JEJU_with_embeddings_donghyeok.csv 로드
        logging.info("JEJU_with_embeddings_donghyeok.csv 로드 시작")
        df = pd.read_csv('./data/JEJU_with_embeddings_donghyeok.csv', encoding='utf-8-sig')
        df['가맹점명'] = df['가맹점명'].str.replace(' ', '')
        
        logging.info(f"데이터 크기: {len(df)}")
        
        # FAISS 인덱스 생성
        location_embeddings = np.stack(df['장소_임베딩'].apply(str_to_array).to_numpy()).astype(np.float32)
        category_embeddings = np.stack(df['카테고리_임베딩'].apply(str_to_array).to_numpy()).astype(np.float32)
        
        faiss.normalize_L2(location_embeddings)
        faiss.normalize_L2(category_embeddings)
        
        index_location = faiss.IndexFlatIP(location_embeddings.shape[1])
        index_category = faiss.IndexFlatIP(category_embeddings.shape[1])
        
        index_location.add(location_embeddings)
        index_category.add(category_embeddings)
        
        return df, index_location, index_category
        
    except Exception as e:
        logging.error(f"데이터 로드 및 처리 중 오류 발생: {e}")
        raise

# 메인 코드에서 사용
tokenizer, embedding_model = load_models()
df, index_location, index_category = load_and_process_data()

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

def display_map(df, next_df=None):
    try:
        # 상위 5개 결과 사용
        df = df.head(5)
        logging.info(f"지도 생성 시작 - 현재 추천 식당 수: {len(df)}")
        
        valid_data = []
        current_coords = []
        
        # 현재 추천 장소들의 좌표 처리
        for idx, row in df.iterrows():
            coords = get_coords_from_address(row['주소'])
            if coords:
                current_coords.append((coords['lat'], coords['lng']))
                valid_data.append({
                    'lat': coords['lat'],
                    'lng': coords['lng'],
                    'name': row['가맹점명'],
                    'address': row['주소'],
                    'category': row['Categories'],
                    'similarity_score': row.get('similarity_score', 'N/A'),
                    'local_usage': row['현지인이용건수비중'],
                    'is_current': True
                })
        
        # 다음 시간대 추천 장소들의 좌표 처리
        next_coords = []
        if next_df is not None and not next_df.empty:
            next_df = next_df.head(5)
            logging.info(f"다음 시간대 추천 식당 수: {len(next_df)}")
            for idx, row in next_df.iterrows():
                coords = get_coords_from_address(row['주소'])
                if coords:
                    next_coords.append((coords['lat'], coords['lng']))
                    valid_data.append({
                        'lat': coords['lat'],
                        'lng': coords['lng'],
                        'name': row['가맹점명'],
                        'address': row['주소'],
                        'category': row['Categories'],
                        'similarity_score': row.get('similarity_score', 'N/A'),
                        'local_usage': row['현지인이용건수비중'],
                        'is_current': False
                    })
        
        if not valid_data:
            logging.error("유효한 좌표가 없습니다.")
            return None
            
        # 모든 좌표의 중심점 계산
        all_lats = [d['lat'] for d in valid_data]
        all_lngs = [d['lng'] for d in valid_data]
        center_lat = sum(all_lats) / len(all_lats)
        center_lng = sum(all_lngs) / len(all_lngs)
        
        # 지도 생성
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=12
        )
        
        # 현재 추천 그룹과 다음 추천 그룹 생성
        current_group = folium.FeatureGroup(name="현재 추천")
        next_group = folium.FeatureGroup(name="다음 시간대 추천")
        
        # 현재 추천 그룹의 클러스터 원 그리기
        if current_coords:
            # 현재 추천 그룹의 중심점 계산
            current_center_lat = sum(lat for lat, _ in current_coords) / len(current_coords)
            current_center_lng = sum(lng for _, lng in current_coords) / len(current_coords)
            
            # 중심점에서 가장 먼 마커까지의 거리 계산 (미터 단위)
            max_distance = 0
            for lat, lng in current_coords:
                distance = geodesic((current_center_lat, current_center_lng), (lat, lng)).meters
                max_distance = max(max_distance, distance)
            
            # 클러스터 원 그리기 (반경을 20% 더 크게)
            folium.Circle(
                location=(current_center_lat, current_center_lng),
                radius=max_distance * 1.2,  # 20% 더 크게
                color='red',
                fill=True,
                fill_opacity=0.1,
                popup='현재 추천 지역'
            ).add_to(current_group)
        
        # 다음 추천 그룹의 클러스터 원 그리기
        if next_coords:
            # 다음 추천 그룹의 중심점 계산
            next_center_lat = sum(lat for lat, _ in next_coords) / len(next_coords)
            next_center_lng = sum(lng for _, lng in next_coords) / len(next_coords)
            
            # 중심점에서 가장 먼 마커까지의 거리 계산 (미터 단위)
            max_distance = 0
            for lat, lng in next_coords:
                distance = geodesic((next_center_lat, next_center_lng), (lat, lng)).meters
                max_distance = max(max_distance, distance)
            
            # 클러스터 원 그리기 (반경을 20% 더 크게)
            folium.Circle(
                location=(next_center_lat, next_center_lng),
                radius=max_distance * 1.2,  # 20% 더 크게
                color='blue',
                fill=True,
                fill_opacity=0.1,
                popup='다음 시간대 추천 지역'
            ).add_to(next_group)
        
        # 마커 추가
        for data in valid_data:
            popup_content = f"""
                <div style='width: 200px'>
                    <b>{data['name']}</b><br>
                    카테고리: {data['category']}<br>
                    주소: {data['address']}<br>
                    유사도: {data['similarity_score']}<br>
                    현지인이용비중: {data['local_usage']}%
                </div>
            """
            
            if data['is_current']:
                folium.Marker(
                    [data['lat'], data['lng']],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(current_group)
            else:
                folium.Marker(
                    [data['lat'], data['lng']],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(next_group)
        
        # 그룹을 지도에 추가
        current_group.add_to(m)
        next_group.add_to(m)
        
        # 레이어 컨트롤 추가
        folium.LayerControl().add_to(m)
        
        logging.info(f"지도 생성 완료 - 총 표시된 식당 수: {len(valid_data)}")
        return m
        
    except Exception as e:
        logging.error(f"지도 생성 중 오류 발생: {str(e)}", exc_info=True)
        return None

def find_restaurants_by_keywords(df, user_keywords):
    """
    사용자 키워드와 일치하는 키워드를 가진 식당들을 찾아 반환합니다.
    """
    try:
        matching_restaurants = []
        
        # 각 식당의 키워드를 확인
        for idx, row in df.iterrows():
            if isinstance(row.get('Keywords'), str):
                try:
                    restaurant_keywords = ast.literal_eval(row['Keywords'])
                except:
                    restaurant_keywords = {}
            else:
                restaurant_keywords = row.get('Keywords', {})
                
            # 사용자 키워드가 식당 키워드에 포함되어 있는지 확인
            if any(keyword in restaurant_keywords for keyword in user_keywords):
                matching_restaurants.append(idx)
                logging.info(f"매칭된 식당: {row['가맹점명']}, 키워드: {restaurant_keywords}")
        
        if not matching_restaurants:
            logging.warning(f"키워드 {user_keywords}와 일치하는 식당을 찾을 수 없습니다.")
            return pd.DataFrame()
            
        return df.loc[matching_restaurants]
        
    except Exception as e:
        logging.error(f"키워드 기반 식당 검색 중 오류 발생: {str(e)}")
        return pd.DataFrame()

def generate_response(prompt, result_df, gender_preference, time, category_type, keywords=None):
    try:
        # 입력값 로깅
        logging.info(f"\n{'='*50}\n응답 생성 시작\n{'='*50}")
        logging.info(f"입력 정보:")
        logging.info(f"- 원본 프롬프트: {prompt}")
        logging.info(f"- 성별 선호: {gender_preference}")
        logging.info(f"- 시간대: {time}")
        logging.info(f"- 카테고리: {category_type}")
        logging.info(f"- 키워드: {keywords}")
        
        if result_df is None or result_df.empty:
            logging.warning("추천할 식당이 없습니다.")
            return "죄송합니다. 조건에 맞는 식당을 찾을 수 없습니다."

        result_df = result_df.head(5)
        restaurant_info = ""
        
        for idx, row in result_df.iterrows():
            info = (f"### 가게 이름: {row['가맹점명']}\n"
                    f"- **주소**: {row['주소']}\n"
                    f"- **업종**: {row['Categories']}\n"
                    f"- **가성비 여부**: {row['가성비여부']}\n"
                    f"- **별점**: {row['별점']}\n"
                    f"- **방문자 리뷰 개수**: {row['방문자리뷰개수']}\n"
                    f"- **영업시간**: {row['영업시간']}\n"
                    f"- **정기휴일**: {row['정기휴일']}\n"
                    f"- **휴일 관련 세부사항**: {row['휴일관련세부사항']}\n"
                    f"- **브레이크타임**: {row['브레이크타임']}\n"
                    f"- **라스트오더타임**: {row['라스트오더타임']}\n"
                    f"- **요약된 리뷰**: {row['요약된리뷰']}\n"
                    f"- **Top-1 키워드**: {row['Top-1_키워드']}\n"
                    f"- **Top-2 키워드**: {row['Top-2_키워드']}\n"
                    f"- **Top-3 키워드**: {row['Top-3_키워드']}\n"
                    f"- **Top-5 메인메뉴**: {row['Top-5_메인메뉴']}\n")
            
            restaurant_info += info + "\n"

        full_prompt = f"""
        당신은 제주도의 맛집을 잘 아는 현지 맛집 전문가입니다. 
        사용자의 선호도와 조건을 고려하여 맛집을 추천해주세요.

        ## 입력 정보
        - **사용자 질문**: {prompt}
        - **선호 시간대**: {time}
        - **선호 성별**: {gender_preference}
        - **선호 업종**: {category_type}
        {f'- **검색 키워드**: {", ".join(keywords)}' if keywords else ''}

        ## 추천 식당 정보
        {restaurant_info}

        ## 답변 형식
        다음 형식을 반드시 지켜서 답변을 작성해주세요:

        1. 첫 문단: 사용자의 요청을 요약하고 추천하는 식당들의 전반적인 특징 소개
           - 존댓말 사용
           - 각 식당 이름은 **볼드체**로 강조
           - 간단한 특징 언급

        2. 구분선 추가: "---"

        3. 각 식당별 상세 설명 (모든 추천 식당에 대해 반복):
           ### 🏆 **[식당이름]**
           
           - 📍 **위치**: [주소]
           - 🕒 **영업 시간**: [영업시간]
           - ⏰ **브레이크타임**: [브레이크타임]
           - 🔚 **라스트오더**: [라스트오더타임]
           - 📅 **정기휴일**: [정기휴일]
           - ⭐ **별점**: [별점]
           - 👥 **방문자 리뷰**: [방문자리뷰개수]건
           
           **대표 메뉴**:
           [Top-5_메인메뉴]
           
           **주요 키워드**:
           - [Top-1_키워드]
           - [Top-2_키워드]
           - [Top-3_키워드]
           
           **리뷰 요약**:
           [요약된리뷰]
           
           **추천 이유**:
           1. **[특징1]**
              상세 설명...
           2. **[특징2]**
              상세 설명...

           ---

        4. 마무리 멘트:
           - 모든 추천 식당들의 공통적인 장점 언급
           - 방문 시 기대할 수 있는 점
           - 긍정적 마무리

        주의사항:
        - 각 식당의 특징은 **볼드체**로 강조
        - 구체적인 수치와 정보 포함
        - 친근하고 전문적인 톤 유지
        - 주어진 정보를 그대로 활용하여 작성(정보가 없다면 해당 내용 출력하지 말 것)
        - 실제 데이터에 없는 내용은 추가하지 말 것
        - 모든 추천 식당에 대해 균형있게 설명할 것
        """
        
        # 토큰 수 계산 및 로깅 추가
        total_tokens = model.count_tokens(full_prompt)
        logging.info(f"입력 토큰 수: {total_tokens}")
        
        logging.info(f"Gemini 프롬프트 입력: {full_prompt}")
        
        # LLM 요청 전 시간 기록
        start_time = time_module.time()
        
        # 향상된 LLM 요청 설정
        response = model.generate_content(
            full_prompt
        )
        
        # 응답 시간 계산
        end_time = time_module.time()
        response_time = end_time - start_time
        
        if not response or not response.text:
            logging.error("LLM 응답이 비어있습니다.")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다."
        
        # 응답 로깅
        logging.info(f"\nLLM 응답 정보:")
        logging.info(f"- 응답 시간: {response_time:.2f}초")
        logging.info(f"- 응답 내용: {response.text[:200]}...")  # 처음 200자만 로깅

        return response.text
        
    except Exception as e:
        logging.error(f"응답 생성 중 오류 발생: {str(e)}")
        return "죄송합니다. 응답 생성 중 오류가 발생했습니다."

def filter_by_keywords(df, keywords, threshold=0.3):
    """
    키워드 기반으로 데이터프레임을 필터링하는 함수
    """
    try:
        logging.info(f"키워드 필터링 시작 - 원본 키워드: {keywords}")
        if not keywords or df.empty:
            return df

        # 키워드 임베딩을 계산
        keyword_embeddings = [embed_text(keyword) for keyword in keywords]
        keyword_embeddings = np.array(keyword_embeddings, dtype=np.float32)
        faiss.normalize_L2(keyword_embeddings)

        filtered_rows = []
        similarities = []

        for idx, row in df.iterrows():
            try:
                # 각 행의 키워드 임베딩을 가져옴
                top1_emb = str_to_array(row['Top-1_키워드_임베딩'])
                top2_emb = str_to_array(row['Top-2_키워드_임베딩'])
                top3_emb = str_to_array(row['Top-3_키워드_임베딩'])

                # nan 값 처리
                if np.isnan(top1_emb).any() or np.isnan(top2_emb).any() or np.isnan(top3_emb).any():
                    logging.error(f"임베딩 변환 오류: {row['가맹점명']}의 임베딩에 nan 값이 포함되어 있습니다.")
                    continue

                # 코사인 유사도 계산
                row_embeddings = np.array([top1_emb, top2_emb, top3_emb], dtype=np.float32)
                faiss.normalize_L2(row_embeddings)

                # FAISS 인덱스 생성 및 검색
                index = faiss.IndexFlatIP(row_embeddings.shape[1])
                index.add(row_embeddings)

                # 유사도 검색
                sim, _ = index.search(keyword_embeddings, 1)
                max_similarity = np.max(sim)

                logging.info(f"행 {idx} - 최대 유사도: {max_similarity:.4f}")

                if max_similarity > threshold:
                    filtered_rows.append(idx)
                    similarities.append(max_similarity)

            except Exception as e:
                logging.error(f"행 {idx} 처리 오류: {str(e)}")
                continue

        if not filtered_rows:
            logging.warning("키워드 매칭 결과 없음")
            return df

        filtered_df = df.loc[filtered_rows].copy()
        filtered_df['keyword_similarity'] = similarities
        filtered_df = filtered_df.sort_values('keyword_similarity', ascending=False)

        logging.info(f"키워드 필터링 결과: {len(filtered_df)}개 항목 매칭")
        return filtered_df

    except Exception as e:
        logging.error(f"키워드 필터링 중 오류 발생: {str(e)}", exc_info=True)
        return df

def filter_data_by_criteria(df, location, category, time_periods=None, conditions=None, gender_preference=None):
    try:
        logging.info(f"\n{'='*50}\n필터링 프로세스 시작\n{'='*50}")
        logging.info(f"입력 파라미터:\n- 위치: {location}\n- 카테고리: {category}\n- 시간대: {time_periods}\n- 조건: {conditions}\n- 성별 선호: {gender_preference}")
        logging.info(f"초기 데이터 크기: {len(df)} 행")
        
        filtered_df = df.copy()
        
        # 1. FAISS 임베딩 기반 검색
        logging.info("1. FAISS 임베딩 기반 검색 시작")
        
        # 문자열을 numpy 열로 변환하는 함수
        def parse_embedding(embedding_str):
            # 문자에서 숫자만 추출
            numbers = [float(num) for num in embedding_str.strip('[]').split()]
            return np.array(numbers)

        try:
            # CSV 파일 읽기 시 인코딩 설정
            if '장소_임베딩' not in filtered_df.columns:
                embedding_column = [col for col in filtered_df.columns if '임베딩' in col][0]
                filtered_df = filtered_df.rename(columns={embedding_column: '장소_임베딩'})
            
            # 데이터프레임의 임베딩을 numpy 열로 변환
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
            time_mapping = {
                '아침': '5시11시이용건수비중',
                '점심': '12시13시이용건수비중',
                '오후': '14시17시이용건수비중',
                '저녁': '18시22시이용건수비중',
                '밤': '23시4시이용건수비중' 
            }
            
            filtered_df['시간대_점수'] = 0
            valid_times = []
            
            for t in time_periods:
                if t in time_mapping:
                    column = time_mapping[t]
                    if column in filtered_df.columns:
                        # 데이터 타입이 숫자형인지 확인하고 변환
                        filtered_df[column] = filtered_df[column].astype(float)
                        filtered_df['시간대_점수'] += filtered_df[column]
                        valid_times.append(t)
                    else:
                        logging.warning(f"컬럼 '{column}'이 데이터프레임에 없습니다.")
            
            if valid_times:
                temp_df = filtered_df[filtered_df['시간대_점수'] > 0]
                if not temp_df.empty:
                    filtered_df = temp_df
                    filtered_df = filtered_df.sort_values('시간대_점수', ascending=False)
                    logging.info(f"시간대 필터링 후 데이터 수: {len(filtered_df)}")
                else:
                    logging.warning("시간대 조건을 만족하는 데이터가 없어 전체 데이터를 유지합니다.")
            else:
                logging.warning("유효한 시간대가 없습니다.")
        
        # 성별 필터링 추가
        if gender_preference:
            before_count = len(filtered_df)
            if isinstance(gender_preference, list):
                gender_preference = gender_preference[0]
            
            # '남성 이용 비중 높음' 또는 '여성 이용 비중 높음' 조건 추가
            if gender_preference.lower() in ["남성", "남성 이용 비중 높음"]:
                filtered_df = filtered_df[filtered_df['최근12개월남성회원수비중'] > filtered_df['최근12개월여성회원수비중']]
            elif gender_preference.lower() in ["여성", "여성 이용 비중 높음"]:
                filtered_df = filtered_df[filtered_df['최근12개월여성회원수비중'] > filtered_df['최근12개월남성회원수비중']]
            
            after_count = len(filtered_df)
            logging.info(f"성별 '{gender_preference}' 필터링 결과: {before_count} -> {after_count} 행")
        
        # 3. 조건 필터링
        if conditions:
            logging.info("3. 추가 조건 필터링 시작")
            for condition in conditions:
                temp_df = filtered_df.copy()
                condition = condition.lower()
                
                # 가성비 여부 조건 추가
                if "가성비" in condition:
                    temp = temp_df[temp_df['가성비여부'] == 1]
                    if not temp.empty:
                        filtered_df = temp

                # 이용 건수 조건
                if "상위 10%" in condition:
                    temp = temp_df[temp_df['이용건수구간'].isin(['5_75~90%', '6_90~100%'])]
                    if not temp.empty:
                        filtered_df = temp
                
                # 현지인 조건
                elif "현지인" in condition:
                    temp_df = temp_df.sort_values('현지인이용건수비중', ascending=False)
                    temp = temp_df.head(max(int(len(temp_df) * 0.2), 1))
                    if not temp.empty:
                        filtered_df = temp
                
                # 연령대 조건
                elif any(age in condition for age in ['20대', '30대', '40대', '50대', '60대']):
                    age_mapping = {
                        '20대': '최근12개월20대이하회원수비중',
                        '30대': '최근12개월30대회원수비중',
                        '40대': '최근12개월40대회원수비중',
                        '50대': '최근12개월50대회원수비중',
                        '60대': '최근12개월60대이상회원수비중'
                    }
                    
                    for age, column in age_mapping.items():
                        if age in condition:
                            # 조건에서 퍼센트 값을 추출
                            percent_match = re.search(r'(\d+)%', condition)
                            threshold = float(percent_match.group(1)) if percent_match else 20.0
                            
                            temp = temp_df[temp_df[column] >= threshold]
                            if not temp.empty:
                                filtered_df = temp
                                logging.info(f"{age} 연령대 {threshold}% 이상 필터링 결과: {len(temp)}개 식당")
                            else:
                                logging.warning(f"{age} 연령대 {threshold}% 이상 조건을 만족하는 식당이 없습니다")
                            break
                
                logging.info(f"조건 '{condition}' 적용 후 데이터 수: {len(filtered_df)}")

        # 각 필터링 단계마다 상세 로깅 추가
        if time_periods:
            for time in time_periods:
                before_count = len(filtered_df)
                # ... 기존 시간대 필터링  ...
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
        
        # 최종 결과 대한 상세 정보 로깅
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
                    elif time == '오후':
                        logging.info(f"- 오후 이용 비중: {row['14시17시이용건수비중']:.2f}%")
                    elif time == '저녁':
                        logging.info(f"- 저녁 이용 비중: {row['18시22시이용건수비중']:.2f}%")
                    elif time == '밤':
                        logging.info(f"- 심야 이용 비중: {row['23시4시이용건수비중']:.2f}%")
        
        return filtered_df.head(5)  # 최종 결과 반환 (최대 5개)

    except Exception as e:
        logging.error(f"필터링 중 오류 발생: {str(e)}", exc_info=True)
        return filtered_df.head(5)  # 오류 발생시 현재까지의 결과 반환

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
    
    prompt = f"""
    제주도 맛집 데이터를 기반으로 사용자 입력을 구조화된 JSON으로 변환하는 작업입니다.
    다음 단계에 따라 분석해주세요:
    1. 장소 정보 추출
    - 제주시/서귀포시 구분
    - 읍/면/동 단위까지 포함
    - 예: "제주시 노형동", "서귀포시 색달동"
    2. 업종 분류
    - 주요 카테고리: "육류,고기요리", "해물,생선요리", "가정식", "단품요리 " 등
    - 체 업종으로 매핑
    3. 시간대 분석
    - 아침(5-11시), 점심(12-13시), 저녁(17-21시), 밤(22-23시)
    - 해당하는 시간대 배열로 표현
    4. 성별 분석
    - 남성 이용 비중이 높은 경우: "남성 이용 비중 높음"
    - 여성 이용 비중이 높은 경우: "여성 이용 비중 높음"
    - 성별 정보가 없는 경우 빈 리스트로 표현
    5. 조건 분석 (아래 조건만 사용 가능)
    - 이용건수: "이용건수 상위 10%", "이용건수 상위 20%"
    - 연령대: 
        "20대 이하 이용 비중 10% 이상", "30대 이용 비중 30% 이상", "50대 이용 비중 20% 이상" 등
    - 요일: "월요일 이용 비중 가장 높음", "화요일 이용 비중 가장 높음" 등
    - 현지인: "현지인 이용 비중 가장 높음"
    - 가성비: "가성비"
    6. 키워드 조건
    - 음식점의 특징을 나타내는 키워드 추출
    - 예: "음식이 맛있어요", "가성비가 좋아요", "양이 많아요", "친절해요" 등
    출력 형식:
    {{
        "장소": "지역명",
        "Categories": "업종",
        "시간대": ["아침", "점심", "저녁", "밤"],
        "성별": ["성별 조건"],
        "조건": ["조건1", "조건2", ...],
        "키워드": ["키워드1", "키워드2", ...]
    }}
    예시 입력/출력:
    1. 입력: "제주시 노형동에서 이용객이 많고 맛있는 가성비 고기집 추천해줘"
    → {{
        "장소": "제주시 노형동",
        "Categories": "육류,고기요리",
        "시간대": [],
        "성별": [],
        "조건": ["이용건수 상위 10%", "가성비"],
        "키워드": ["음식이 맛있어요", "고기 질이 좋아요"]
    }}
    2. 입력: "서귀포시 중문동에서 현지인들이 자주가고 주말에 인기있는 해산물집 알려줘"
    → {{
        "장소": "서귀포시 중문동",
        "Categories": "해물,생선요리",
        "시간대": [],
        "성별": [],
        "조건": ["현지인 이용 비중 가장 높음", "토요일 이용 비중 가장 높음"],
        "키워드": ["음식이 맛있어요", "재료가 신선해요"]
    }}
    3. 입력: "나는 가장인데 제주시 연동에서 저녁에 가족들과 깔끔한 한식집 추천해줘"
    → {{
        "장소": "제주시 연동",
        "Categories": "가정식",
        "시간대": ["저녁"],
        "성별": ["남성 이용 비중 높음"],
        "조건": ["50대 이용 비중 10% 이상", "이용건수 상위 20%"],
        "키워드": ["매장이 청결해요", "가족모임 하기 좋아요"]
    }}
    4. 입력: "점심에 여자인 친구들이랑 서귀포시 색달동에서 20대가 많이 가고 금요일에 인기있는 분위기 좋은 맛집 알려줘"
    → {{
        "장소": "서귀시 색달동",
        "Categories": "단품요리 전문",
        "시간대": ["점심"],
        "성별": ["여성 이용 비중 높음"],
        "조건": ["20대 이용 비중 30% 이상", "금요일 이용 비중 가장 높음", "가성비"],
        "키워드": ["인테리어가 멋져요", "분위기가 좋아요"]
    }}
    5. 입력: "제주시 이도동에서 점심에 남성 직장인이 많이 가는 맛집 추천해줘"
    → {{
        "장소": "제주시 이도동",
        "Categories": "단품요리 전문",
        "시간대": ["점심"],
        "성별": ["남성 이용 비중 높음"],
        "조건": ["30대 이용 비중 20% 이상", "이용건수 상위 10%"],
        "키워드": ["혼밥하기 좋아요", "음식이 맛있어요"]
    }}
    6. 입력: "애월읍에서 현지인이 추천하는 인기많은 오션뷰 맛집 알려줘"
    → {{
        "장소": "제주시 애월읍",
        "Categories": "단품요리 전문",
        "시간대": [],
        "성별": [],
        "조건": ["현지인 이용 비중 가장 높음", "이용건수 상위 10%"],
        "키워드": ["뷰가 좋아요", "음식이 맛있어요"]
    }}
    7. 입력: "제주시 구좌읍에서 아침일찍 문여는 신선한 해산물집 중에 인기있는 곳 추천해줘"
    → {{
        "장소": "제주시 구좌읍",
        "Categories": "해물,생선요리",
        "시간대": ["아침"],
        "성별": [],
        "조건": ["이용건수 상위 20%", "현지인 이용 비중 가장 높음"],
        "키워드": ["재료가 신선해요", "식이 있어요"]
    }}
    8. 입력: "서귀포시 대정읍에서 30대가 많이 가고 회식하기 좋은 식당 알려줘"
    → {{
        "장소": "서귀포시 대정읍",
        "Categories": "단품요리 전문",
        "시간대": ["저녁"],
        "성별": [],
        "조건": ["30대 이용 비중 30% 이상", "이용건수 상위 20%"],
        "키워드": ["단체모임 하기 좋아요", "매장이 넓어요"]
    }}
    9. 입력: "제주시 화북동에서 주말에 인기있고 가족들이 자주가는 밥집 추천해줘"
    → {{
        "장소": "제주시 화북동",
        "Categories": "가정식",
        "시간대": [],
        "성별": [],
        "조건": ["50대 이용 비중 30% 이상", "토요일 이용 비중 가장 높음"],
        "키워드": ["차하기 편해요", "가족모임 하기 좋아요"]
    }}
    10. 입력: "서귀포시 서귀동에서 밤늦게까지 하고 젊은층이 여자들이 많이 가는 맛있는 고기집 알려줘"
    → {{
        "장소": "서귀포시 서귀동",
        "Categories": "육류,고기요리",
        "시간대": ["밤"],
        "성별": ["여성 이용 비중 높음"],
        "조건": ["20대 이용 비중 30% 이상", "이용건수 상위 10%", "가성비"],
        "키워드": ["음식이 맛있어요", "고기 질이 좋아요"]
    }}
    현재 입력: {user_input}
    위 예시들을 참고하여 JSON 형식으로만 출력해주세요.
    """
    
    try:
        logging.info(f"\n{'='*50}\nLLM 요청 시작\n{'='*50}")
        logging.info(f"사용자 입력: {user_input}")
        
        total_tokens_prompt = model.count_tokens(prompt)
        logging.info(f"입력 토큰 수: {total_tokens_prompt}")

        # 타임아웃 설정 추가
        response = model.generate_content(
            prompt,
        )
        
        if not response or not response.text:
            logging.error("LLM 응답이 비어있습니다.")
            return None
            
        json_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        logging.info(f"LLM 응답: {json_str}")
        
        parsed_input = eval(json_str)
        if "장소" in parsed_input and "Categories" in parsed_input:
            logging.info(f"파싱 성공: {parsed_input}")
            return parsed_input
            
    except Exception as e:
        logging.error(f"JSON 파싱 실패: {str(e)}")
        return None

# 채 기록 초기화 함수
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "어떤 식당을 찾으시나요?"}]
    logging.info("채팅 기록 초기화")

def get_next_time_period(current_time):
    """현재 시간대를 기반으로 다음 추천할 시간대를 반환"""
    time_flow = {
        "아침": "점심",
        "점심": "오후",
        "오후": "저녁",
        "저녁": "밤"
    }
    return time_flow.get(current_time)

def suggest_next_destination(current_df, time_periods, location):
    """다음 시간대의 주변 맛집을 추천하는 함수"""
    try:
        # 입력값 유효성 검사 추가
        if current_df is None or current_df.empty:
            logging.warning("현재 추천 데이터가 비어있습니다.")
            return None, None
            
        if not time_periods:  # 시간대가 빈 리스트인 경우
            logging.warning("시간대 정보가 없습니다.")
            return None, None
            
        current_time = time_periods[0]  # 현재 시간대
        next_time = get_next_time_period(current_time)
        
        if not next_time:  # 다음 시간대가 없는 경우
            logging.warning(f"'{current_time}' 다음 시간대가 없습니다.")
            return None, None
            
        logging.info(f"다음 동선 추천 시작 - 현재 시간대: {current_time}, 다음 시간대: {next_time}")
        
        # 현재 추천된 식당들의 중심 좌표 계산
        center_coords = None
        for _, row in current_df.head().iterrows():
            coords = get_coords_from_address(row['주소'])
            if coords:
                if center_coords is None:
                    center_coords = coords
                else:
                    center_coords['lat'] = (center_coords['lat'] + coords['lat']) / 2
                    center_coords['lng'] = (center_coords['lng'] + coords['lng']) / 2
        
        if not center_coords:
            return None, None
            
        # 주변 식당 필터링을 위한 임시 데이터프레임 생성
        filtered_df = filter_data_by_criteria(
            df=df,  # 전역 데이터프레임
            location=location,  # 현재 위치
            category=None,  # 카테고리 제한 없음
            time_periods=[next_time],  # 다음 시간대
            conditions=None,
            gender_preference=None
        )
        
        if filtered_df is None or filtered_df.empty:
            return None, None
            
        # 응답 생성
        next_destination_response = f"""
        ### 🕒 다음 {next_time} 시간대 추천 맛집

        현재 위치 근처에서 {next_time}에 방문하기 좋은 맛집을 추천해드립니다:
        """
        
        # 새로운 Gemini 프롬프트 생성
        prompt = f"""
        다음 시간대({next_time})에 방문하기 좋은 주변 맛집들의 특징을 요약해서 추천해주세요.
        
        추천할 식당 정보:
        {filtered_df.head().to_dict('records')}
        
        다음 형식으로 작성해주세요:
        1. 전반적인 특징 소개
        2. 각 식당별 핵심 특징 (1-2줄)
        3. 마무리 멘트
        """
        
        response = model.generate_content(prompt)
        if response and response.text:
            next_destination_response += response.text
            
        return next_destination_response, filtered_df
        
    except Exception as e:
        logging.error(f"다음 동선 추천 중 오류 발생: {str(e)}")
        return None, None

# Streamlit UI 설정
st.title("🍊 제주 맛집 추천 챗봇")

# 세션 상태 초기화 부분
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "어떤 식당을 찾으시나요?"}]
    
if "map_data" not in st.session_state:
    st.session_state.map_data = None

# 상태 변수 추가
if 'current_recommendation' not in st.session_state:
    st.session_state.current_recommendation = None
if 'next_recommendation' not in st.session_state:
    st.session_state.next_recommendation = None

# 사용자 입력 처리 부분
if prompt := st.chat_input("예: 아침에 20대인 남자인 친구들과 애월에 해산물집을 갈 거야"):
    try:
        logging.info("\n" + "="*50)
        logging.info("새로운 사용자 요청 처리 시작")
        logging.info(f"사용자 입력: {prompt}")

        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

        parsed_input = parse_user_input_to_json(prompt)
        logging.info(f"파싱된 JSON 결과: {parsed_input}")

        if parsed_input:
            location = parsed_input["장소"]
            category = parsed_input["Categories"]
            time = parsed_input.get("시간대", [])
            conditions = parsed_input.get("조건", [])
            keywords = parsed_input.get("키워드", [])
            gender_preference = parsed_input.get("성별", [])

            # 기본 필터링
            result_df = filter_data_by_criteria(df, location, category, time_periods=time, conditions=conditions, gender_preference=gender_preference)

            # 키워드 기반 필터링
            if keywords and not result_df.empty:
                result_df = filter_by_keywords(result_df, keywords)

            # 첫 번째 응답 저장
            st.session_state.current_recommendation = {
                'response': generate_response(
                    prompt=prompt,
                    result_df=result_df,
                    gender_preference=gender_preference,
                    time=", ".join(time) if time else "",
                    category_type=category,
                    keywords=keywords
                ),
                'df': result_df.copy()  # DataFrame 복사본 저장
            }

            # 다음 추천 준비
            if time and result_df is not None and not result_df.empty:
                next_suggestion_result = suggest_next_destination(result_df, time, location)
                if next_suggestion_result is not None:
                    next_suggestion, next_df = next_suggestion_result
                    if next_suggestion and next_df is not None and not next_df.empty:
                        st.session_state.next_recommendation = {
                            'response': next_suggestion,
                            'df': next_df.copy()
                        }
                    else:
                        st.session_state.next_recommendation = None
                else:
                    st.session_state.next_recommendation = None
            else:
                st.session_state.next_recommendation = None

            # 응답을 메시지 히스토리에 추가
            with st.chat_message("assistant"):
                st.markdown(st.session_state.current_recommendation['response'])
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": st.session_state.current_recommendation['response']
                })

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

# 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 추천 결과 표시 (채팅 UI 외부)
if st.session_state.current_recommendation:
    st.markdown("---")
    st.markdown("## 🍽️ 맛집 추천 결과")
    
    # 탭 생성
    tab1, tab2 = st.tabs(["📍 맛집 지도", "🕒 다음 시간대 추천"])
    
    # 첫 번째 탭: 현재 추천 (다음 시간대 정보 포함)
    with tab1:
        if st.session_state.current_recommendation['df'] is not None:
            next_df = None
            if st.session_state.next_recommendation and st.session_state.next_recommendation['df'] is not None:
                next_df = st.session_state.next_recommendation['df']
            
            current_map = display_map(
                st.session_state.current_recommendation['df'],
                next_df
            )
            if current_map:
                st_folium(current_map, width=700, height=500)
    
    # 두 번째 탭: 다음 시간대 추천 (지도 없이 텍스트만)
    with tab2:
        if st.session_state.next_recommendation:
            st.markdown(st.session_state.next_recommendation['response'])
        else:
            st.info("다음 시간대 추천이 없습니다.")

# 사이드바 설정
st.sidebar.markdown("### 🗺️ 지도 설명")
st.sidebar.markdown("""
- 🔴 **빨간색**: 현재 추천 맛집
- 🔵 **파란색**: 다음 시간대 추천 맛집
    - *다음 시간대 추천은 시간 정보가 있는 경우에만 표시됩니다*
""")

# 구분선 추가
st.sidebar.markdown("---")

# 초기화 버튼
if st.sidebar.button('🗑️ 대화 기록 지우기'):
    st.session_state.messages = [{"role": "assistant", "content": "어떤 식당을 찾으시나요?"}]
    st.session_state.current_recommendation = None
    st.session_state.next_recommendation = None
    st.experimental_rerun()


