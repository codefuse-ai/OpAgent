import json
import os
from PIL import Image
import requests
from datetime import datetime
from io import BytesIO
import base64
import uuid
import fcntl


# import html2text


def load_jsonl_to_list(jsonl_file_path):
    data_list = []
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            data_list.append(json_obj)
    return data_list


def load_dataset_from_file(filename):
    if filename.endswith('.json'):
        with open(filename, 'r') as file:
            return json.load(file)
    elif filename.endswith('.jsonl'):
        return load_jsonl_to_list(filename)
    else:
        raise ValueError("Invalid file format. Please provide a .json or .jsonl file.")


def save_dataset(data, filename, convert_to_jsonl=True):
    with open(filename, 'w', encoding='utf-8', errors='ignore') as file:
        # 加文件锁防止并发冲突
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            if convert_to_jsonl:
                if isinstance(data, list):
                    for obj in data:
                        try:
                            record = json.dumps(obj, default=str, ensure_ascii=False)
                            file.write(record + '\n')
                        except Exception as e:
                            print(f"记录丢弃：{e}")
                else:
                    try:
                        record = json.dumps(data, default=str, ensure_ascii=False)
                        file.write(record + '\n')
                    except Exception as e:
                        print(f"记录丢弃：{e}")
            else:
                # 分页写入大JSON
                file.write('[\n')
                page_size = 1000
                for i in range(0, len(data), page_size):
                    page = data[i:i + page_size]
                    json_str = json.dumps(page, default=str, ensure_ascii=False)[1:-1]
                    if i > 0:
                        file.write(',\n')
                    file.write(json_str)
                file.write('\n]')
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def read_txt_from_file(file_path):
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                lines.append(line)
        print(f"成功读取文件，共 {len(lines)} 行")
        return lines
    except:
        print(f"读取错误: {file_path}")
        return []


def find_type_files(directory, file_type=None):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file_type or file.endswith(file_type):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    return file_list


def load_save_image(source, save_directory, filename=None):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)
    if filename is None:
        time_now = datetime.now().strftime('%Y%m%d%H%M%S')
        uid = uuid.uuid4()
        filename = f"screenshot_{time_now}_{uid}.png"
    file_path = os.path.join(save_directory, filename)

    if os.path.exists(source):
        image = Image.open(source)
    elif source.startswith(('http://', 'https://')):
        response = requests.get(source)
        image = Image.open(BytesIO(response.content))
    elif source.startswith(('data:image/', 'iVBOR')):
        if 'data:image/' in source:
            base64_data = source.split('base64,')[1]
        else:
            base64_data = source
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data))
    else:
        raise ValueError("图片类型不支持")

    image.save(file_path)
    return file_path


def load_save_html(url, save_directory, filename=None):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        if not filename:
            time_now = datetime.now().strftime('%Y%m%d%H%M%S')
            uid = uuid.uuid4()
            filename = f"page_{time_now}_{uid}.html"
        save_file_name = os.path.join(save_directory, filename)

        with open(save_file_name, 'w', encoding='utf-8') as f:
            f.write(response.content.decode())
        # print(f"文件已保存到：{save_path}")
        return save_file_name
    except Exception as e:
        print(f"下载失败：{str(e)}")
        return None

# def html_to_markdown(url):
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#     except requests.exceptions.RequestException as e:
#         return f"Error fetching URL: {str(e)}"

#     soup = BeautifulSoup(response.content, 'html.parser')
#     main_content = soup.find('main') or soup.find('article') or soup.body

#     converter = html2text.HTML2Text()
#     converter.ignore_links = False
#     converter.ignore_images = False
#     converter.wrap_links = False  # 防止链接换行

#     markdown = converter.handle(str(main_content))
#     clean_md = "\n".join([line.strip() for line in markdown.split("\n") if line.strip()])

#     return clean_md