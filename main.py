import csv
import math

data = './data/'
LOG_LENGTH_RANGE = 0.1
DISTANCE_THRESHOLD = 0.9  # TODO this is a placeholder, research and change accordingly

keywords = ['error', 'warning', 'application', 'service']


class Cluster:

    def __init__(self, log=None, keyword=None):
        self.logs = [log]
        self.keyword = keyword
        if log is not None:
            self.log_template = self.create_log_template(log)

    def create_log_template(self, log):
        return log

    def add_log_to_cluster(self, log):
        self.logs.append(log)

    def update_log_template(self, log):
        return None


def distance(template, log):
    return 0


def get_most_similar_cluster(clusters, log):
    """
    Returns the cluster with the minimum distance between its log template and the given log
    :param clusters: the list of clusters (will be the clusters returned from get_similar_clusters)
    :param log: the log message
    :return: the most similar cluster
    """
    minimum_distance = -1
    most_similar_cluster = None
    for c in clusters:
        dist = distance(c.log_template, log)
        if minimum_distance < 0 or dist < minimum_distance:
            minimum_distance = dist
            most_similar_cluster = c
    return most_similar_cluster


def add_log_to_keyword_clusters(clusters, log):
    for c in clusters:
        if log.contains(c.keyword):
            c.add_log_to_clusters(log)


def create_keyword_clusters():
    """
    Creates the list of clusters based on keywords (e.g., 'Error', 'Warning')
    :return: list of keyword clusters
    """
    keyword_clusters = []
    for k in keywords:
        keyword_clusters.append(Cluster(keyword=k))
    return keyword_clusters


def get_similar_clusters(clusters, log):
    """
    Finds all clusters that are similar to the given log. Clusters are similar if:
        1) The log's length is within 10% of the cluster's log template's length
        2) The similarity score calculated between the cluster's log template and the given log exceeds a
        predetermined threshold
    :param clusters: the list of clusters to search from
    :param log: the log
    :return: list of similar clusters
    """
    similar_clusters = []
    for c in clusters:
        template = c.log_template
        len(template.split())
        # Check if log's length equals template's length +- 10%
        len_template = len(template.split())
        min_len = math.floor(len_template - (len_template * LOG_LENGTH_RANGE))
        max_len = math.ceil(len_template + (len_template * LOG_LENGTH_RANGE))
        if len(log.split()) in range(min_len, max_len):
            # Check if similarity is less than threshold
            if distance(template, log) < DISTANCE_THRESHOLD:
                similar_clusters.append(c)
    return similar_clusters


def create_clusters(reader):
    """
    Creates the log and keyword clusters
    :param reader: Reader for the CSV file mapping field names to values
    :return: two lists containing the log and keyword clusters, respectively
    """
    log_clusters = []  # Clusters based on log similarity
    keyword_clusters = create_keyword_clusters()  # Clusters based on keywords
    for row in reader:
        log = row['Content']
        add_log_to_keyword_clusters(keyword_clusters, log)
        similar_clusters = get_similar_clusters(log_clusters, log)
        # If there's no similar clusters, then create a new one with the log
        if not similar_clusters:
            new_cluster = Cluster(log=log)
            log_clusters.append(new_cluster)
        else:
            most_similar_cluster = get_most_similar_cluster(similar_clusters, log)  # Find most similar cluster
            most_similar_cluster.add_log_to_cluster(log)  # Add new log to most similar cluster
            most_similar_cluster.update_log_template(log)  # Update most similar cluster's log template

    return keyword_clusters, log_clusters


# Open the file, call tokenize() to create lists of tokens, log tokens, and log token lengths
def process_file(file_name):
    print('Processing file ', file_name)
    filepath = data + file_name
    with open(filepath) as csv_file:
        reader = csv.DictReader(csv_file)
        create_clusters(reader)


if __name__ == '__main__':
    process_file('Windows_2k.log_structured.csv')