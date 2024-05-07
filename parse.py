import argparse
import json
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation

# Function to load the conversation file
def load_conversations(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Function to extract user messages from the conversation data
def extract_user_messages(conversations):
    user_messages = []
    for conversation in conversations:
        mapping = conversation.get('mapping', {})
        for node in mapping.values():
            message = node.get('message', None)
            if message and message.get('author', {}).get('role') == 'user':
                content_parts = message.get('content', {}).get('parts', [])
                if isinstance(content_parts, list) and content_parts:
                    user_messages.append(content_parts[0])
    return user_messages

# List of the top 100 most common English words
top_english_words = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", "not", "on", "with", "he",
    "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about",
    "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than",
    "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two",
    "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us"
]

# Print Top English Words
def print_english_words(count=100):
    print(f"\nTop {count} Most Common English Words:\n", top_english_words[:count])

# Common Words Analysis with Option to Filter English Words
def common_words_analysis(messages, top_n=20, filter_english=False):
    words = [word for message in messages if isinstance(message, str) for word in message.lower().split()]
    if filter_english:
        words = [word for word in words if word not in top_english_words]
    common_words = Counter(words).most_common(top_n)
    print(f"\nTop {top_n} Common Words (Filtered={filter_english}):\n", common_words)

# Topic Modeling Analysis
def topic_modeling_analysis(messages, num_topics=10, top_words=10):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', lowercase=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform([msg for msg in messages if isinstance(msg, str)])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf_matrix)
    words = tfidf_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words_list = [words[i] for i in topic.argsort()[:-top_words-1:-1]]
        print(f"\nTopic {topic_idx}: {top_words_list}")

# Question Rephrasing Analysis
def question_rephrasing_analysis(messages, similarity_threshold=0.5):
    questions = [msg for msg in messages if isinstance(msg, str) and msg.strip().endswith('?')]
    vectorizer = CountVectorizer().fit_transform(questions)
    similarity_matrix = cosine_similarity(vectorizer)
    similar_questions = [(questions[i], questions[j]) for i in range(len(questions)) for j in range(i+1, len(questions)) if similarity_matrix[i, j] > similarity_threshold]
    print("\nSimilar Question Pairs (Threshold={}):\n".format(similarity_threshold), similar_questions[:5])

# CLI Argument Parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform various analyses on conversation data.')
    parser.add_argument('file', help='Path to the conversation JSON file')
    parser.add_argument('--common-words', action='store_true', help='Perform Common Words Analysis')
    parser.add_argument('--topic-modeling', action='store_true', help='Perform Latent Topic Modeling')
    parser.add_argument('--rephrasing', action='store_true', help='Perform Question Rephrasing Analysis')
    parser.add_argument('--print-english-words', type=int, nargs='?', const=100, help='Print the top N most common English words')
    parser.add_argument('--filter-english-words', action='store_true', help='Filter out top English words from common words analysis')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top results for common words')
    parser.add_argument('--num-topics', type=int, default=10, help='Number of topics for topic modeling')
    parser.add_argument('--top-words', type=int, default=10, help='Number of words per topic for topic modeling')
    parser.add_argument('--similarity-threshold', type=float, default=0.5, help='Threshold for question rephrasing similarity')

    args = parser.parse_args()
    conversations = load_conversations(args.file)
    user_messages = extract_user_messages(conversations)

    if args.print_english_words:
        print_english_words(args.print_english_words)
    if args.common_words:
        common_words_analysis(user_messages, args.top_n, args.filter_english_words)
    if args.topic_modeling:
        topic_modeling_analysis(user_messages, args.num_topics, args.top_words)
    if args.rephrasing:
        question_rephrasing_analysis(user_messages, args.similarity_threshold)

