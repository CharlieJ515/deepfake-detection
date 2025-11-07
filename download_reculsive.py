import requests
import subprocess
import os
import xml.etree.ElementTree as ET

# --- 1. 사용자 설정 (이전과 동일) ---
USERNAME = "deepfake_detection"
PASSWORD = "deepfake_cse340"
WEBDAV_URL = "https://cloud.kyusang-jang.com/remote.php/dav/files/deepfake_detection/data/"

# ---------------------------------

def get_file_list(url, auth):
    """WebDAV 서버에 PROPFIND 요청을 보내 파일 목록을 가져옵니다."""
    headers = {'Depth': 'infinity'}  # 재귀적으로 모든 하위 폴더 탐색
    try:
        print(f"서버에 파일 목록을 요청합니다: {url}")
        response = requests.request('PROPFIND', url, headers=headers, auth=auth)
        response.raise_for_status()  # 오류가 있으면 예외 발생
        
        print("파일 목록 응답을 받았습니다. 파싱을 시작합니다...")
        return parse_xml_for_files(response.content)
        
    except requests.exceptions.RequestException as e:
        print(f"[오류] 서버 요청 실패: {e}")
        return None

def parse_xml_for_files(xml_content):
    """WebDAV (XML) 응답을 파싱하여 파일 경로만 추출합니다."""
    # XML 네임스페이스
    namespaces = {'d': 'DAV:'}
    tree = ET.fromstring(xml_content)
    
    file_paths = []
    base_path = "/remote.php/dav/files/deepfake_detection/data/"
    
    for response in tree.findall('.//d:response', namespaces):
        href = response.find('./d:href', namespaces).text
        
        # href에서 기본 경로 제거하여 상대 경로 추출
        if href.startswith(base_path) and href != base_path:
            relative_path = href[len(base_path):]
            
            # 폴더가 아닌 파일만 추가 (URL 디코딩 및 슬래시 제거)
            if not relative_path.endswith('/'):
                file_paths.append(relative_path)
                
    print(f"총 {len(file_paths)}개의 파일 경로를 찾았습니다.")
    return file_paths

def download_file(file_path, auth):
    """단일 파일을 wget으로 다운로드합니다."""
    full_url = f"{WEBDAV_URL}{file_path}"
    local_file_path = f"./data/{file_path}" # 현재 위치에 data 폴더를 만들고 그 안에 저장
    
    # 이 파일을 저장할 로컬 폴더 생성
    local_dir = os.path.dirname(local_file_path)
    os.makedirs(local_dir, exist_ok=True)
        
    # wget 명령어 생성
    command = f'wget --user "{auth[0]}" --password "{auth[1]}" "{full_url}" -O "{local_file_path}"'
    
    print("\n" + "="*50)
    print(f"다운로드 중: {file_path}")
    print(f"저장 위치: {local_file_path}")
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"  > 성공: {file_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  > [실패!] {file_path} 다운로드 중 오류 발생.")
        return False

# --- 메인 스크립트 실행 ---
if __name__ == "__main__":
    auth = (USERNAME, PASSWORD)
    file_list = get_file_list(WEBDAV_URL, auth)
    
    if file_list:
        total = len(file_list)
        print(f"\n총 {total}개의 파일 다운로드를 시작합니다.")
        
        success_count = 0
        for i, file_path in enumerate(file_list):
            print(f"\n--- 파일 {i+1}/{total} ---")
            if download_file(file_path, auth):
                success_count += 1
            else:
                print("스크립트를 중단합니다.")
                break
        
        print("\n" + "="*50)
        print(f"다운로드 완료: 총 {total}개 중 {success_count}개 성공")
    else:
        print("다운로드할 파일을 찾지 못했거나 서버 연결에 실패했습니다.")