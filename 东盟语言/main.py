import json
import re
import pandas as pd

def chinese_sent_tokenize(text):
    # 使用正则表达式分割中文句子，基于句号、感叹号、问号和换行
    sentences = re.split(r'(?<=。|！|？|\n)\s*', text)
    return [s.strip() for s in sentences if s.strip()]

def read_xlsx_file(file_path):
    """
    使用pandas读取xlsx文件，提取文章数据。
    假设xlsx文件有列：id, title, publish_time, source, content, search_keyword, created_at
    返回一个列表，每个元素是字典：{'id': id, 'content': content, 'search_keyword': search_keyword}
    """
    df = pd.read_excel(file_path, sheet_name=0)  # 读取第一个sheet
    articles = []
    for _, row in df.iterrows():
        article = {
            'id': str(row.get('id', '')),  # 转换为字符串，确保一致性
            'content': str(row.get('content', '')).strip(),
            'search_keyword': str(row.get('search_keyword', '未知')).strip()
        }
        # 移除truncated标记，如果存在
        article['content'] = re.sub(r'\(truncated \d+ characters\)\.\.\.', '', article['content'])
        articles.append(article)
    return articles

def classify_label(keyword):
    """
    根据search_keyword分类到docx领域的标签。
    """
    keyword_lower = keyword.lower()
    if any(word in keyword_lower for word in ['水稻', '稻田', '超级稻', '粮食作物', '经济作物', '园艺作物']):
        return '种植业'
    elif any(word in keyword_lower for word in ['畜产品加工', '猪', '养殖', '肉鸡', '疫病防控', '饲料营养']):
        return '畜牧业'
    elif any(word in keyword_lower for word in ['水产', '渔业', '水产品加工', '海洋捕捞']):
        return '渔业'
    elif any(word in keyword_lower for word in ['政策', '经济', '法律法规', '市场分析', '国际贸易', '乡村振兴']):
        return '农业政策与经济'
    elif any(word in keyword_lower for word in ['科技', '智慧农业', '生物技术', '农业机械化']):
        return '农业科技'
    else:
        return '未知'

def split_and_label_articles(articles):
    """
    对每个文章的内容进行句子分割，并打标签。
    返回句子列表，每个是字典。
    """
    sentenced_data = []
    for article in articles:
        content = article['content']
        label = classify_label(article['search_keyword'])
        sentences = chinese_sent_tokenize(content)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 5 or not (50 <= len(sent) <= 80):  # 过滤不符合要求的句子
                continue
            length = len(sent)  # 中文字数
            sentenced_data.append({
                'sentence': sent,
                'label': label,
                'original_article_id': article['id'],
                'length': length
            })
    return sentenced_data

# 主程序
if __name__ == '__main__':
    file_path = '命中关键词文章.xlsx'  # 假设文件名为此，根据实际附件调整
    articles = read_xlsx_file(file_path)  # 使用pandas读取xlsx文件
    sentenced_data = split_and_label_articles(articles)  # 分割并打标
    
    # 输出到JSON文件
    with open('labeled_sentences.json', 'w', encoding='utf-8') as f:
        json.dump(sentenced_data, f, ensure_ascii=False, indent=4)
    
    # 打印前5个示例
    print("前5个句子示例:")
    for item in sentenced_data[:5]:
        print(item)