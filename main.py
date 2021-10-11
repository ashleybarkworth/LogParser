import csv
import hashlib
import os
import re

import textdistance
import pandas as pd
import dataset_settings
import evaluator

data_dir = './data/'
output_dir = './output/'
DISTANCE_THRESHOLD = 0.1

keywords = ['error', 'warning', 'application', 'service']
c_id = 0


class Cluster:

    def __init__(self, keyword=None):
        global c_id
        self.id = c_id
        c_id += 1

        self.log_template = None
        self.logs = []
        self.ids = []
        self.keyword = keyword
        self.count = 1

    def add_log_to_cluster(self, log, i):
        self.count += 1
        self.logs.append(log)
        self.ids.append(i)
        # If it's not a keyword cluster then update template
        if self.keyword is None:
            self.update_log_template(log)

    def update_log_template(self, log):
        if self.log_template is None:
            self.log_template = log
        else:
            log_tokens = log.split()
            self.log_template = ' '.join([token if token == log_tokens[idx]
                                          else '<*>' for idx, token in enumerate(self.log_template.split())])

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

        logs = 'LOGS ({0} total))\n'.format(len(self.logs))
        for i, log in enumerate(self.logs):
            logs += 'Log {0}: '.format(i) + log + '\n'

        return delimiter + title + template_or_keyword + logs + delimiter


def levenshteinDistance(template, log):
    dist2 = textdistance.levenshtein(template, log)
    return dist2 / max(len(template), len(log))


def add_log_to_keyword_clusters(clusters, log, i):
    """
    Checks to see if log message contains any of the keywords. If so, the log is added to the corresponding cluster.
    :param clusters: list of keyword clusters
    :param log: the log message
    """
    for c in clusters:
        # Convert to uppercase in order to search for all possible cases, e.g. 'error', 'Error', 'ERROR'
        if c.keyword.upper() in log.upper():
            c.add_log_to_cluster(log, i)


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
            # Check if distance is less than threshold
            distance = levenshteinDistance(filled_template, log)
            if distance < DISTANCE_THRESHOLD:
                candidates.append((c, distance))
    return candidates


def preprocess_log(log, regex):
    for currentRex in regex:
        log = re.sub(currentRex, '<*>', log)
    return log


def create_clusters(reader, regex):
    """
    Creates the log and keyword clusters
    :param reader: Reader for the CSV file mapping field names to values
    :param regex:
    :return: two lists containing the log and keyword clusters, respectively
    """
    log_clusters = []  # Clusters based on log similarity
    keyword_clusters = create_keyword_clusters()  # Clusters based on keywords
    templates = []  # Stores 'ground truth' templates for each log
    for i, row in enumerate(reader):
        log = row['Content']
        log = preprocess_log(log, regex)
        add_log_to_keyword_clusters(keyword_clusters, log, i)
        candidates = get_similar_clusters(log_clusters, log)
        # If there's no similar clusters, then create a new one with the log
        candidates.sort(key=lambda c: c[1], reverse=True)
        if len(candidates) > 0:
            most_similar_cluster = candidates[0][0]
            most_similar_cluster.add_log_to_cluster(log, i)  # Add new log to most similar cluster
        else:
            new_cluster = Cluster()
            new_cluster.add_log_to_cluster(log, i)
            log_clusters.append(new_cluster)
        templates.append(row['EventTemplate'])

    return keyword_clusters, log_clusters, templates


def print_clusters(log_clusters, keyword_clusters):
    print('Keyword Clusters\n==============\n')
    [print(c) for c in log_clusters]
    print('Log Clusters\n==============\n')
    [print(c) for c in keyword_clusters]


def write_results(log_clusters, templates, file_name):
    df_log = pd.DataFrame()
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    df_event = []
    templates = [0] * len(templates)
    template_ids = [0] * len(templates)
    for c in log_clusters:
        template = c.log_template
        eventid = hashlib.md5(' '.join(template).encode('utf-8')).hexdigest()[0:8]
        logids = c.ids
        for logid in logids:
            templates[logid] = template
            template_ids[logid] = eventid
        df_event.append([eventid, template, len(logids)])

    df_log['EventId'] = template_ids
    df_log['EventTemplate'] = templates

    pd.DataFrame(df_event, columns=['EventId', 'EventTemplate', 'Occurrences']).to_csv(
        os.path.join(output_dir, file_name + '_templates.csv'), index=False)
    df_log.to_csv(os.path.join(output_dir, file_name + '_structured.csv'), index=False)


def calculate_accuracy(filename):
    accuracy = evaluator.evaluate(
        groundtruth=os.path.join(data_dir, filename + '_structured.csv'),
        parsedresult=os.path.join(output_dir, filename + '_structured.csv')
    )
    print('Parsing Accuracy: {:.4f}'.format(accuracy))


# Open the file, call tokenize() to create lists of tokens, log tokens, and log token lengths
def process_file(settings):
    file_name = settings['log_file']
    filepath = data_dir + file_name + '_structured.csv'
    regex = settings['regex']
    with open(filepath) as csv_file:
        reader = csv.DictReader(csv_file)
        keyword_clusters, log_clusters, templates = create_clusters(reader, regex)
        # print_clusters(log_clusters, keyword_clusters)
        write_results(log_clusters, templates, file_name)
        calculate_accuracy(file_name)


if __name__ == '__main__':
    for dataset, settings in dataset_settings.settings.items():

        print('Processing dataset {0}\n========================'.format(dataset))
        process_file(settings)
        print('========================\n')

