from string import atof,atoi
def parse_class_report(class_report) :
    labels = []
    precision = []
    recall = []
    f1 = []
    support = []
    results = [s.split() for s in class_report.split('\n')]
    for i in range(2,len(results)-3) :
        result = results[i]
        labels.append(result[0])
        precision.append(atof(result[1]))
        recall.append(atof(result[2]))
        f1.append(atof(result[3]))
        support.append(atoi(result[4]))
    return (labels,precision,recall,f1,support)
