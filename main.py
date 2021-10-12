import csv
import hashlib
import os

import pandas as pd
import textdistance

import dataset_settings
import evaluator

data_dir = './data/'
templates_dir = './templates/'
json_dir = './json/'
DISTANCE_THRESHOLD = 0.1

keywords = ['error', 'warning', 'application', 'service']


class Cluster:

    def __init__(self, keyword=None):
        self.log_template = None
        self.logs = []
        self.log_ids = []
        self.keyword = keyword
        self.count = 1

    def add_log_to_cluster(self, log, i):
        self.count += 1
        self.logs.append(log)
        self.log_ids.append(i)
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
        title = 'CLUSTER \n'
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
    :param i: the line number
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


def create_clusters(reader):
    """
    Creates the log and keyword clusters
    :param reader: Reader for the CSV file mapping field names to values
    :return: two lists containing the log and keyword clusters, respectively
    """
    log_clusters = []  # Clusters based on log similarity
    keyword_clusters = create_keyword_clusters()  # Clusters based on keywords
    templates = []  # Stores 'ground truth' templates for each log
    for i, row in enumerate(reader):
        log = row['Content']
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


def write_results(log_clusters, templates, filename):
    df_log = pd.DataFrame()
    if not os.path.isdir(templates_dir):
        os.makedirs(templates_dir)

    clusters = []
    templates = [0] * len(templates)
    template_ids = [0] * len(templates)
    for c in log_clusters:
        template = c.log_template
        event_id = hashlib.md5(' '.join(template).encode('utf-8')).hexdigest()[0:8]
        log_ids = c.log_ids
        for log_id in log_ids:
            templates[log_id] = template
            template_ids[log_id] = event_id
        clusters.append([template, len(log_ids)])

    df_log['EventId'] = template_ids
    df_log['EventTemplate'] = templates

    df_templates = pd.DataFrame(clusters, columns=['Log Template', 'Number of Occurrences'])
    df_templates.to_csv(os.path.join(templates_dir, filename + '_templates.csv'), index=False)
    df_log.to_csv(os.path.join(templates_dir, filename + '_structured.csv'), index=False)


def calculate_accuracy(filename):
    accuracy = evaluator.evaluate(
        groundtruth=os.path.join(data_dir, filename + '.csv'),
        parsedresult=os.path.join(templates_dir, filename + '_structured.csv')
    )
    print('Parsing Accuracy: {:.4f}'.format(accuracy))
    return accuracy

# Open the file, create clusters, compute accuracy, and write convert structured logs to JSON
def process_file(settings):
    filename = settings['log_file']
    filepath = data_dir + filename + '.csv'

    dataNewDataframe = []  # Storing all the value to required from CSV file to convert into JSON
    logsClusterDict = {}  # Create dictionary for logs -> clusterId
    logsTemplateDict = {}  # Create dictionary for logs -> LogTemplate
    with open(filepath) as csv_file:
        reader = csv.DictReader(csv_file)
        keyword_clusters, log_clusters, templates = create_clusters(reader)
        print('Number of clusters: {0}'.format(len(log_clusters)))
        # print_clusters(log_clusters, keyword_clusters)
        write_results(log_clusters, templates, filename)
        # Accuracy calculation
        calculate_accuracy(filename)
        # JSON conversion
        # print(len(keyword_clusters))
        # print(len(log_clusters))
        for clusterID in range(len(log_clusters)):
            for logID in range(len(log_clusters[clusterID].logs)):
                logsClusterDict[log_clusters[clusterID].logs[logID]] = clusterID + 1
                logsTemplateDict[log_clusters[clusterID].logs[logID]] = log_clusters[clusterID].log_template

    # print(logsTemplateDict)
    # Reading file again to prepare final JSON output
    with open(filepath) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            log = row['Content']
            clusterId = logsClusterDict[log]
            logTemplate = logsTemplateDict[log]

            if 'Date' in row:
                date = row['Date']
            else:
                date = '-'
            dataNewDataframe.append({'Line ID': row['LineId'],
                                     'Structured Log': logTemplate,
                                     'Date': date,
                                     'Time': row['Time'],
                                     'Content': row['Content'],
                                     'Belongs to which cluster': clusterId, })

    newDataframe = pd.DataFrame(dataNewDataframe)
    newDataframe.stack().unstack(0).reset_index()

    if not os.path.isdir(json_dir):
        os.makedirs(json_dir)
    json_file = json_dir + filename + '.json'
    newDataframe.to_json(json_file, orient='records', lines=True)


if __name__ == '__main__':
    for dataset, settings in dataset_settings.settings.items():
        print('Processing dataset {0}\n========================'.format(dataset))
        process_file(settings)
        print('')
