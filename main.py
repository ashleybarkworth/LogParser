import csv
import math
import re

import numpy
import textdistance

data = './data/'
DISTANCE_THRESHOLD = 0.3  # TODO this is a placeholder, research and change accordingly

keywords = ['error', 'warning', 'application', 'service']
c_id = 0


class Cluster:

    def __init__(self, log=None, keyword=None):
        global c_id
        self.id = c_id
        c_id += 1
        self.logs = [log] if log is not None else []
        self.keyword = keyword
        self.count = 1
        if log is not None:
            self.log_template = log

    def add_log_to_cluster(self, log):
        self.count += 1
        self.logs.append(log)

    def update_log_template(self, log):
        log_tokens = log.split()
        self.log_template = ' '.join([token if token == log_tokens[idx] else '<*>' for idx, token in enumerate(self.log_template.split())])

    def fill_wildcards(self, log):
        """
        Fills the template's wildcards with given log values for more accurate comparison
        e.g., if self.log_template = "Error id *: service * terminated unexpectedly",
                and log = "Error id 984: service 9213 terminated unexpectedly", then
                filled_template = "Error id 984: service 9213 terminated unexpectedly"
        :param log: log containing values for wildcards
        :return: log template with wildcards replaced with log's values, if any wildcards are present
        """
        log_tokens = log.split()
        filled_template = ' '.join([token if token != '<*>' else log_tokens[idx]
                                   for idx, token in enumerate(self.log_template.split())])
        return filled_template

    def __str__(self):
        delimiter = '--------------------------------------------\n'
        title = 'CLUSTER ID {0}\n'.format(self.id)
        if hasattr(self, 'log_template'):
            template_or_keyword = 'TEMPLATE)\n{0}\n'.format(self.log_template)
        elif hasattr(self, 'keyword'):
            template_or_keyword = 'KEYWORD)\n{0}\n'.format(self.keyword)
        else:
            return AttributeError

        logs = 'LOGS)\n'
        for i, log in enumerate(self.logs):
            logs += 'Log {0}: '.format(i) + log + '\n'

        return delimiter + title + template_or_keyword + logs + delimiter


def levenshteinDistance(template, log):
    #
    # distances = numpy.zeros((len(template) + 1, len(log) + 1))
    # # Initializing the distance matrix
    # for templateMatrix in range(len(template) + 1):
    #     distances[templateMatrix][0] = templateMatrix
    #
    # for logMatrix in range(len(log) + 1):
    #     distances[0][logMatrix] = logMatrix
    #
    # # iterate loop through each cell in matrix
    # for templateMatrix in range(1, len(template) + 1):
    #     for logMatrix in range(1, len(log) + 1):
    #
    #         if template[templateMatrix - 1] == log[logMatrix - 1]:
    #             distances[templateMatrix][logMatrix] = distances[templateMatrix - 1][logMatrix - 1]
    #         else:
    #             deletion = distances[templateMatrix][logMatrix - 1]
    #             insertion = distances[templateMatrix - 1][logMatrix]
    #             substitution = distances[templateMatrix - 1][logMatrix - 1]
    #
    #             if deletion <= insertion and deletion <= substitution:
    #                 distances[templateMatrix][logMatrix] = deletion + 1
    #             elif insertion <= deletion and insertion <= substitution:
    #                 distances[templateMatrix][logMatrix] = insertion + 1
    #             else:
    #                 distances[templateMatrix][logMatrix] = substitution + 1
    #
    # # printDistances(distances, len(template), len(log))
    # distance = distances[len(template)][len(log)]
    # Levenshtein distance
    dist2 = textdistance.levenshtein(template, log)

    # print(distance)
    return dist2 / max(len(template), len(log))


def add_log_to_keyword_clusters(clusters, log):
    """
    Checks to see if log message contains any of the keywords. If so, the log is added to the corresponding cluster.
    :param clusters: list of keyword clusters
    :param log: the log message
    """
    for c in clusters:
        # Convert to uppercase in order to search for all possible cases, e.g. 'error', 'Error', 'ERROR'
        if c.keyword.upper() in log.upper():
            c.add_log_to_cluster(log)


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
    :return: list of similar clusters with their distances
    """
    candidates = []
    for c in clusters:
        template = c.log_template
        # For now this just checks if log lengths are equal, maybe look at comparing different lengths later
        if len(log.split()) == len(template.split()):
            filled_template = c.fill_wildcards(log)
            # Check if similarity is less than threshold
            distance = levenshteinDistance(filled_template, log)
            if distance < DISTANCE_THRESHOLD:
                candidates.append((c, distance))
    return candidates


def create_clusters(reader):
    """
    Creates the log and keyword clusters
    :param reader: Reader for the CSV file mapping field names to values
    :return: two lists containing the log and keyword clusters, respectively
    """
    log_clusters = []  # Clusters based on log similarity
    keyword_clusters = create_keyword_clusters()  # Clusters based on keywords
    for row in reader:
        # log = sanitize_input(row['Content'], file_type)
        log = row['Content']
        add_log_to_keyword_clusters(keyword_clusters, log)
        candidates = get_similar_clusters(log_clusters, log)
        # If there's no similar clusters, then create a new one with the log
        candidates.sort(key=lambda c: c[1], reverse=True)
        if len(candidates) > 0:
            most_similar_cluster = candidates[0][0]
            most_similar_cluster.add_log_to_cluster(log)  # Add new log to most similar cluster
            most_similar_cluster.update_log_template(log)  # Update most similar cluster's log template
        else:
            new_cluster = Cluster(log=log)
            log_clusters.append(new_cluster)

    return keyword_clusters, log_clusters


def print_clusters(clusters):
    [print(c) for c in clusters]


# Open the file, call tokenize() to create lists of tokens, log tokens, and log token lengths
def process_file(file_name):
    print('Processing file ', file_name)
    filepath = data + file_name
    dataNewDataframe = [] # Storing all the value to required from CSV file to convert into JSON
    logsClusterDict = {} #Create dictionary for logs -> clusterId
    with open(filepath) as csv_file:
        reader = csv.DictReader(csv_file)
        keyword_clusters, log_clusters = create_clusters(reader)
        print(len(keyword_clusters))
        print(len(log_clusters))
        for clusterID in range(len(log_clusters)):
            for logID in range(len(log_clusters[clusterID].logs)):
                logsClusterDict[log_clusters[clusterID].logs[logID]] = clusterID+1

    # Reading file again to prepare final JSON output            
    with open(filepath) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            log = row['Content']
            clusterId = logsClusterDict[log]
            dataNewDataframe.append({'Line ID':row['LineId'], 
                                     'Structured Log':log ,
                                     'Date': row['Date'],
                                     'Time':row['Time'],
                                     'Content':row['Content'],
                                     'Belongs to which cluster':clusterId,})
            
    newDataframe = pd.DataFrame(dataNewDataframe)
    #print(newDataframe)
    newDataframe.to_json('demo.json', orient='records', lines=True)
   
        # Print out keyword and log clusters
        # print('Keyword Clusters\n==============\n')
        # print_clusters(keyword_clusters)
        # print('Log Clusters\n==============\n')
        # print_clusters(log_clusters)
    print('Done clustering')


if __name__ == '__main__':
    process_file('Thunderbird_2k.log_structured.csv')
