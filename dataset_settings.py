settings = {
    'HDFS': {
        'log_file': 'HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'CT': 0.35,
        'lowerBound': 0.25,
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']
        },

    'Hadoop': {
        'log_file': 'Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'CT': 0.4,
        'lowerBound': 0.2,
        'regex': [r'(\d+\.){3}\d+']
        },

    'Spark': {
        'log_file': 'Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'CT': 0.35,
        'lowerBound': 0.3,
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+']
        },

    'Zookeeper': {
        'log_file': 'Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'CT': 0.4,
        'lowerBound': 0.7,
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?']
        },

    'BGL': {
        'log_file': 'BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'CT': 0.4,
        'lowerBound': 0.01,
        'regex': [r'core\.\d+']
        },

    'HPC': {
        'log_file': 'HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'CT': 0.58,
        'lowerBound': 0.25,
        'regex': [r'=\d+']
        },

    'Thunderbird': {
        'log_file': 'Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'CT': 0.3,
        'lowerBound': 0.2,
        'regex': [r'(\d+\.){3}\d+']
        },

    'Windows': {
        'log_file': 'Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'CT': 0.3,
        'lowerBound': 0.25,
        'regex': [r'0x.*?\s']
        },

    'Linux': {
        'log_file': 'Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'CT': 0.3,
        'lowerBound': 0.3,
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}']
        },

    'Andriod': {
        'log_file': 'Andriod_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'CT': 0.25,
        'lowerBound': 0.3,
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b']
        },

    'HealthApp': {
        'log_file': 'HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'CT': 0.25,
        'lowerBound': 0.3,
        'regex': []
        },

    'Apache': {
        'log_file': 'Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'CT': 0.3,
        'lowerBound': 0.4,
        'regex': [r'(\d+\.){3}\d+']
        },

    'Proxifier': {
        'log_file': 'Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'CT': 0.9,
        'lowerBound': 0.25,
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
        },

    'OpenSSH': {
        'log_file': 'OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'CT': 0.78,
        'lowerBound': 0.25,
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+']
        },

    'OpenStack': {
        'log_file': 'OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'CT': 0.9,
        'lowerBound': 0.25,
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+']
    },

    'Mac': {
        'log_file': 'Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'CT': 0.3,
        'lowerBound': 0.25,
        'regex': [r'([\w-]+\.){2,}[\w-]+']
        }
}