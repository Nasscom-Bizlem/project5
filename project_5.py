import requests
import threading
import json
import pandas as pd
import numpy as np
import re
import bs4
import traceback


MIN_NUMBER_OF_WORDS = 5
VOTE_RATE = 0.3
MAX_THREADS = 50
URL_CHAIN = 'http://35.188.227.39:8080/enhancer/chain/scorpiosvchain'
LABEL_CHAIN = 'http://fise.iks-project.eu/ontology/entity-label'
TYPE_CHAIN = 'http://fise.iks-project.eu/ontology/entity-type'

def strip(s):
    return ''.join(re.split('[^a-zA-Z0-9]', s.lower()))

def get_start_of_word(s):
    it = re.finditer('[a-zA-Z0-9,\./\+]+', s)
    return [ i.start() for i in it ]

def split_row(row, pos):
    r = []
    for i in range(0, len(pos) - 1):
        r.append(row[pos[i]:pos[i + 1]])
    r.append(row[pos[-1]:])
    return r

def is_null_row(row):
    return row.isnull().sum() == len(row)

errors = []
def request_word(word):
    try:
        r = requests.post(URL_CHAIN, data=str(word).encode('utf-8'), headers={'Content-Type': 'application/pdf'})
        r = r.json()

        res = []
        track = []
        for obj in r:
            if LABEL_CHAIN in obj:
                v = obj[LABEL_CHAIN][0]['@value']
                if strip(v) not in track:
                    res.append(v)
                    track.append(strip(v))
        return res
    except Exception as e:
        traceback.print_exc()
        errors.append(word)
        return []

def p5_process_html_no_header(arr, sarr, verbose=True):
    res = [None] * len(arr)
    def request_no_header(arr, i, res):
        r = request_word(arr[i])
        if len(r) > 0:
            res[i] = r


    if verbose: print('requesting in process html no header')

    for i in range(0, len(arr), MAX_THREADS):
        threads = [ threading.Thread(
            target=request_no_header, 
            args=(arr, i, res),
        ) for i in range(i, min(i + MAX_THREADS, len(arr))) ]

        for thread in threads: thread.start()
        for thread in threads: thread.join()

    if verbose: print('finish requesting in process html no header')

    result = []

    for i in range(len(arr)):
        if res[i] is None:
            continue

        pos = [ re.search(word, arr[i], re.IGNORECASE) for word in res[i] ]
        pos = [ (i.start(), i.end()) for i in pos if i is not None]
        pos.sort(key=lambda x: x[0])

        item = []
        cur = 0
        start = 0
        j = 0

        while j < len(arr[i]):
            if arr[i][j] == ' ' and (cur >= len(pos) or j < pos[cur][0]):
                if len(strip(arr[i][start:j])) > 0:
                    item.append(arr[i][start:j])
                start = j + 1
            elif cur < len(pos) and j == pos[cur][0]:
                item.append(arr[i][pos[cur][0]:pos[cur][1]])
                j = pos[cur][1]
                start = j + 1
                cur += 1
            j += 1

        if start < len(arr[i]):
            item.append(arr[i][start:])

        if len(item) >= 12 or len(item) <= 2:
            continue

        obj = {
            'data': {},
            'header': {},
            'line_index': i,
            'StructureType': 'paragraph',
        }

        for i in range(len(item)):
            obj['header'][str(i)] = 'unknown_' + str(i)
            obj['data'][str(i)] = item[i]

        result.append(obj)

    return result

def p5_process_html_with_header(arr, sarr, verbose=True):
    # find headers
    headers = []
    header_index = None

    for i, t in enumerate(sarr):
        if len(t) >= MIN_NUMBER_OF_WORDS:
            headers = t
            header_index = i
            break

    index = header_index + 1
    col_start = [0] * 200

    data_rows = []
    while index < len(arr):
        if re.search('[^=\- ]', arr[index]) is not None and len(sarr[index]) >= MIN_NUMBER_OF_WORDS - 2:
            start = get_start_of_word(arr[index])
            # print(index, sarr[index], start)
            if len(start) / len(headers) > 0.5:
                for p in start:
                    col_start[p] += 1
                data_rows.append(index)

        index += 1

    # process header
    pos = [ k for k, v in enumerate(col_start) if v / len(data_rows) > VOTE_RATE ]
    hpos = get_start_of_word(arr[header_index])
    temp = [ i for i in pos if i in hpos or i + 1 in hpos or i - 1 in hpos ]
    if 0 not in temp:
        temp = [0] + temp
    pos = temp

    headers = [ i.strip() for i in split_row(arr[header_index], pos) ]

    temp = []
    c = 0
    for h in headers:
        if len(h) == 0:
            temp.append('unknown_' + str(c))
            c += 1
        else:
            temp.append(h)
    headers = temp

    if verbose: print(headers)

    result = []
    for i in data_rows:
        obj = {}
        row = [ r.strip() for r in split_row(arr[i], pos) ]

        for j in range(min(len(headers), len(row))):
            obj[headers[j]] = row[j]

        obj['StructureType'] = 'table'
        result.append(obj)

    return result 


def p5_process_html_table(tables, verbose=True):
    container = {}
    threads = [ threading.Thread(
        target=process_table, 
        args=(table, table_name, container, True, verbose)
    ) for table_name, table in tables.items() ]

    # Processing tables...
    if verbose: print('processing tables...')
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if verbose: print('finish processing tables')

    return container



def format_html_table(raw_tables):
    for table in raw_tables:
        height, width = table.shape
        for col in table.columns:
            new_col = table[col].apply(lambda x: None if isinstance(x, str) \
                and len(x) > 0 \
                and (ord(x[0]) < 32 or ord(x[0]) > 126)  \
                else x)
            table[col] = new_col

        for ih in range(height):
            current_value = table.iloc[ih, 0]
            for iw in range(1, width):
                if table.iloc[ih, iw] == current_value:
                    table.iloc[ih, iw] = np.nan
                else:
                    current_value = table.iloc[ih, iw]



def p5_process_html(path, verbose=True):
    with open(path) as f:
        soup = bs4.BeautifulSoup(f, features='lxml')
        for t in soup(['script', 'style', 'meta']):
            t.extract()

        pretty_soup_str = soup.text
        pretty_soup_str = re.sub('\s+<span', '<span', pretty_soup_str)

    total_tables = 0
    try:
        raw_tables = pd.read_html(pretty_soup_str)
        format_html_table(raw_tables)
        tables = separate_tables(raw_tables, table_margin=1)
        r_table = p5_process_html_table(tables, verbose=verbose)
        total_tables = len(raw_tables)
    except ValueError:
        r_table = {}
        if verbose:
            print('No table found')

    soup = bs4.BeautifulSoup(pretty_soup_str, features='lxml')
    for t in soup(['table']):
        t.extract()

    arr = soup.text.replace('\xa0', ' ').split('\n')
    arr = [ t for t in arr if len(t.strip()) > 0 ]
    sarr = [ re.split('\s{2,}', t) for t in arr ]

    found_table = False
    for i, line in enumerate(arr):
        if len(line) < 5 or len(sarr[i]) <= 3:
            continue

        if re.search('[^=\-\s]', line) is None:
            found_table = True
            break

    if not found_table:
        r_table['table_' + str(total_tables)] = {
            'header_data': {},
            'table_data': p5_process_html_no_header(arr, sarr, verbose=verbose),
        }
        return r_table

    r_table['table_' + str(total_tables)] = {
        'header_data': {},
        'table_data': p5_process_html_with_header(arr, sarr, verbose=verbose),
    }

    return r_table


def process_table(table, table_name, container, get_header_data=False, verbose=True):
    table.dropna(how='all', inplace=True, axis=0)
    table.dropna(how='all', inplace=True, axis=1)

    header_index = 0
    height, width = table.shape

    header_data = {}
    while header_index < height:
        c_nan = table.iloc[header_index].isna().sum()
        if c_nan / width < 0.7:
            break

        if get_header_data:
            arr = list(table.iloc[header_index])
            arr = [ str(it) for it in arr if not pd.isnull(it) ]

            r = requests.post(
                URL_CHAIN, 
                data=' '.join(arr).encode('utf-8'), 
                headers={'Content-Type': 'application/pdf'},
            )
            r = r.json()

            for obj in r:
                if LABEL_CHAIN in obj and TYPE_CHAIN in obj:
                    ref = obj[TYPE_CHAIN][0]['@id']

                    if ref not in header_data:
                        header_data[ref] = obj[LABEL_CHAIN][0]['@value']

        header_index += 1

    if get_header_data:
        # process header_data
        selected_headers = set([
            'RepositionRegion', 
            'CargoType', 
            'VesselType',
            'Date',
        ])

        temp_header_data = {}
        for header_key, header_value in header_data.items():
            key = header_key.split('#')[-1]
            if key == 'Port':
                temp_header_data[header_key.replace('Port', 'RepositionRegion')] = header_value
            elif key in selected_headers:
                temp_header_data[header_key] = header_value
        header_data = temp_header_data


    if header_index >= height: return

    columns = table.iloc[header_index].astype(str).tolist()

    # set unknown labels
    c = 0
    for i in range(0, width):
        if pd.isnull(columns[i]) or columns[i] == 'nan':
            columns[i] = 'unknown_' + str(c)
            c += 1

    table.columns = columns
    table.drop(index=table.index[:header_index + 1], inplace=True)


    if verbose:
        print(table.columns)

    results = table.astype(str).to_dict(orient='index')
    res = list(results.values())
    for obj in res:
        for key, value in obj.items():
            if value is None or value == 'nan':
                obj[key] = ''

    separated_res = []
    for i, obj in enumerate(res):
        new_obj = {
            'StructureType': 'table',
            'data': {},
            'header': {},
            'line_index': int(table.iloc[i].name),
        }

        index = 0
        for key, value in obj.items():
            new_obj['header'][index] = key
            new_obj['data'][index] = value
            index += 1

        separated_res.append(new_obj)

    if get_header_data:
        container[table_name] = {
            'header_data': header_data,
            'table_data': separated_res,
        }
    else:
        container[table_name] = res


def separate_tables(raw_tables, table_margin=2):
    count = 0
    tables = {}
    for table in raw_tables:
        height, width = table.shape

        irow = 0
        istart = 0
        iend = 0

        while irow < height:
            while irow < height and is_null_row(table.iloc[irow]):
                irow += 1
            istart = irow

            while irow < height:
                while irow < height and (not is_null_row(table.iloc[irow])):
                    irow += 1
                iend = irow

                while irow < height and is_null_row(table.iloc[irow]):
                    irow += 1

                if irow - iend < table_margin:
                    irow += 1
                else: 
                    iend = irow 
                    break

            if iend - istart >= table_margin:
                tables['table_' + str(count)] = table.iloc[istart:irow].copy()
                count += 1

    return tables


def p5_process_excel(path, verbose=True):
    raw_tables = []

    xls = pd.ExcelFile(path)
    sheets = xls.book.sheets()

    # read visible sheet only
    for sheet in sheets:
        if sheet.visibility == 0:
            table = pd.read_excel(xls, sheet.name, header=None)
            th, tw = table.shape

            if th > 0 and tw > 0:
                raw_tables.append(table)

    tables = separate_tables(raw_tables)

    container = {}
    threads = [ threading.Thread(
        target=process_table, 
        args=(table, table_name, container, True, verbose)
    ) for table_name, table in tables.items() ]

    # Processing tables...
    if verbose:
        print('processing tables...')

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    if verbose:
        print('finish processing tables')


    return container


def process_pdf(path):
    MARGIN_Y = 1.0
    A = 100000

    def get_x(obj):
        return obj['position']['x']

    def get_y(obj):
        return obj['position']['y']

    with open(path) as f:
        data = json.load(f)

    # Build histogram
    y = [ get_y(p) for p in data ]
    y.sort()

    cur_y = y[0]
    hist_y = { cur_y: cur_y }

    for yc in y:
        if yc == cur_y:
            continue
        if yc - cur_y > MARGIN_Y:
            cur_y = yc
        hist_y[yc] = cur_y

    data.sort(key=lambda obj: hist_y[get_y(obj)] * A + get_x(obj))
    slist = {}
    stext = {}
    sarr = {}
    for obj in data:
        hy = hist_y[get_y(obj)]
        if hy not in slist:
            slist[hy] = []
            stext[hy] = []

        slist[hy].append(obj)
        stext[hy].append(obj['text'])

    for line, words in stext.items():
        sentence = ''.join(words)
        sarr[line] = [ word for word in re.split('\s{2,}', sentence) if len(word) > 0 ]

    #for line, words in sarr.items():
    #    print(line, words)

    res = {}
    for line in sarr.keys():
        res[line] = []

    def request_label(sarr, line, index, res):
        r = request_word(sarr[line][index])
        if len(r) > 0:
            res[line].append(r[0])

    def request_row(sarr, line, res):
        threads = [ threading.Thread(target=request_label, args=(sarr, line, index, res)) for index in range(len(sarr[line])) ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    threads = [ threading.Thread(target=request_row, args=(sarr, line, res)) for line in sarr.keys() ]

    print('requesting')
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    result = []
    header_index = None
    for line, words in sarr.items():
        if len(words) <= 3:
            continue

        if len(res[line]) == len(sarr[line]):
            header_index = line
            continue

        is_row = False
        for word in words:
            if re.search('[a-zA-Z0-0]', word) is not None:
                is_row = True
                break

        if not is_row:
            continue

        if header_index is None:
            obj = { 'StructureType': 'paragraph' }
            count = 0
            for word in words:
                obj['unknown_' + str(count)] = word
                count += 1
        else:
            obj = { 'StructureType': 'table' }
            for i in range(min(len(sarr[header_index]), len(words))):
                obj[sarr[header_index][i]] = words[i]

        result.append(obj)

    return result

def p5_process_file(path, verbose=True):
    ext = path.rsplit('.', 1)[1].lower()

    if ext == 'html':
        return p5_process_html(path, verbose=verbose)
    elif ext == 'xls' or ext == 'xlsx':
        return p5_process_excel(path)
    elif ext == 'json':
        return process_pdf(path)


if __name__ == '__main__':
    r = p5_process_html('../data/p5materials/html/c24.html', verbose=True)
    #r = process_pdf('p5materials/pdf/p2.json')
    # r = p5_process_excel('p5materials/excel/x3.xlsx')
    print(json.dumps(r, indent=2))

    # print(errors)