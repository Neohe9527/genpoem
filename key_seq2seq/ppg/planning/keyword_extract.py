"""TextRank 关键词提取"""
import jieba
import jieba.analyse


def extract_keywords(query, num_keywords=4):
    """从用户输入中提取关键词"""
    keywords = jieba.analyse.textrank(query, topK=num_keywords, withWeight=False)
    if not keywords:
        keywords = jieba.analyse.extract_tags(query, topK=num_keywords)
    return keywords
