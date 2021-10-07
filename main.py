import csv

data = './data/'


def tokenize(reader):
    tokens = []  # A list containing all the tokens in the file (e.g., ["first","log","second","log"]
    # A list of lists. Each inner list contains the tokens of one log message (e.g., [["first","log"],["second","log"]]
    tokenized_logs = []
    # A list of lists. Each inner list contains the token lengths of one log message (e.g., [[5,3],[6,3]]
    tokenized_log_lengths = []
    for row in reader:
        content = row['Content']
        print(content)
        # Split log content into tokens based on whitespace
        log_tokens = content.split()
        # Add list of tokens to tokenized_logs
        tokenized_logs.append(log_tokens)
        # Get list containing lengths of each token, then add to tokenized_log_lengths
        log_token_lengths = [len(token) for token in log_tokens]
        tokenized_log_lengths.append(log_token_lengths)
        # Add all tokens to tokens list
        tokens.extend(log_tokens)

    return tokens, tokenized_logs, tokenized_log_lengths


# For each log message in the file, create a list of word tokens and list of word length tokens.
def process_file(file_name):
    filepath = data + file_name
    with open(filepath) as csv_file:
        reader = csv.DictReader(csv_file)
        tokens, tokenized_logs, tokenized_log_lengths = tokenize(reader)
        print('done')


if __name__ == '__main__':
    process_file('Windows_2k.log_structured.csv')


