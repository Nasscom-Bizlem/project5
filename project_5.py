import requests
import threading
import json
import pandas as pd
import numpy as np
import re
import bs4
import traceback
from svg_parser import simplify_json


MIN_NUMBER_OF_WORDS = 5
VOTE_RATE = 0.3
MAX_THREADS = 50
URL_CHAIN = 'http://35.188.227.39:8080/enhancer/chain/scorpiosvchain'
LABEL_CHAIN = 'http://fise.iks-project.eu/ontology/entity-label'
TYPE_CHAIN = 'http://fise.iks-project.eu/ontology/entity-type'
SITE_KEY = 'http://stanbol.apache.org/ontology/entityhub/entityhub#site'

def strip(s):
    return ''.join(re.split('[^a-zA-Z0-9]', s.lower()))

def get_start_of_word(s):
    it = re.finditer('[a-zA-Z0-9,\./\+\$\-]+', s)
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
        r = requests.post(
            URL_CHAIN, 
            data=str(word).encode('utf-8'), 
            headers={'Content-Type': 'application/pdf'},
        )

        r = r.json()

        res = []
        header_data = {}
        count = 0
        for obj in r:
            if SITE_KEY in obj and obj[SITE_KEY][0]['@value'] == 'scorpioheaders':
                count += 1

            if LABEL_CHAIN in obj:
                v = obj[LABEL_CHAIN][0]['@value']
                res.append(v)

                if TYPE_CHAIN in obj:
                    ref = obj[TYPE_CHAIN][0]['@id']
                    if ref not in header_data:
                        header_data[ref] = v

        if count >= 4:
            return res, header_data, True

        return res, header_data, False
    except Exception as e:
        traceback.print_exc()
        errors.append(word)
        return []

def request_header_data(s):
    header_data = {}

    try:
        r = requests.post(
            URL_CHAIN, 
            data=s.encode('utf-8'), 
            headers={'Content-Type': 'application/pdf'},
        )
        r = r.json()

        for obj in r:
            if LABEL_CHAIN in obj and TYPE_CHAIN in obj:
                ref = obj[TYPE_CHAIN][0]['@id']

                if ref not in header_data:
                    header_data[ref] = obj[LABEL_CHAIN][0]['@value']
    except Exception as e:
        traceback.print_exc()

    return header_data

def request_header_data_async(s, index, res):
    res[index] = request_header_data(s)

def p5_process_html_with_lines(arr, sarr, res, r_table, table_name='table', verbose=True):
    if verbose: print('process html with lines')
    result = {
        'header_data': [],
        'table_data': [],
    }

    header_index = 0
    for i in range(len(arr)):
        if len(res[i].keys()) == 0: continue 

        header_item = {
            'line_index': i,
            'number_of_words': len(res[i].keys()),
            'data': [],
        }

        for header_data_key, header_data_value in res[i].items():
            header_item['data'].append({
                'url': header_data_key,
                'word': header_data_value,
            })

        result['header_data'].append(header_item)

        if len(sarr[i]) >= MIN_NUMBER_OF_WORDS:
            header_index = i

        obj = {
            'StructureType': 'line',
            'data': {
                '0': arr[i],
            },
            'header': {
                '0': arr[header_index],
            },
            'line_index': i,
            'label_line_index': header_index,
            'label_string': arr[header_index],
        }

        result['table_data'].append(obj)

    r_table[table_name] = result
    return result


def p5_process_html_no_header(arr, sarr, res, verbose=True):
    if verbose: print('process html with no header')
    result = {
        'header_data': [],
        'table_data': [],
    }

    for i in range(len(arr)):
        if len(res[i].keys()) == 0: continue

        header_item = {
            'line_index': i,
            'number_of_words': len(res[i].keys()),
            'data': [],
        }

        for header_data_key, header_data_value in res[i].items():
            header_item['data'].append({
                'url': header_data_key,
                'word': header_data_value,
            })

        result['header_data'].append(header_item)

        pos = [ re.search(word, arr[i], re.IGNORECASE) for word in res[i].values() ]
        pos = [ (i.start(), i.end()) for i in pos if i is not None ]
        pos.sort(key=lambda x: x[0])

        item = []
        cur, start, j = 0, 0, 0

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

        # if len(item) >= 12 or len(item) <= 2:
        #     continue

        obj = {
            'data': {},
            'header': {},
            'line_index': i,
            'StructureType': 'paragraph',
            'label_line_index': 'unknown',
            'label_string': 'unknown',
        }

        for i in range(len(item)):
            obj['header'][str(i)] = 'unknown_' + str(i)
            obj['data'][str(i)] = item[i]

        result['table_data'].append(obj)

    return result

def p5_process_html_with_header(
    arr, 
    sarr, 
    res, 
    r_table, 
    table_name='table', 
    start_index=0,
    stop_index=None,
    verbose=True,
):
    if stop_index is None:
        stop_index = len(arr)

    if verbose: print('process html with header')
    # find headers
    result = {
        'header_data': [],
        'table_data': [],
    }

    headers = []
    header_index = None

    for i, t in enumerate(sarr[start_index:stop_index]):
        if len(t) >= MIN_NUMBER_OF_WORDS:
            headers = t
            header_index = i + start_index
            break
        else:
            header_item = {
                'line_index': i + start_index,
                'number_of_words': len(res[i].keys()),
                'data': [],
            }

            for header_data_key, header_data_value in res[i + start_index].items():
                header_item['data'].append({
                    'url': header_data_key,
                    'word': header_data_value,
                })

            result['header_data'].append(header_item)

    if header_index is None: return
    index = header_index
    col_start = [0] * 200

    data_rows = []
    while index < min(stop_index, len(arr)):
        if re.match('^[^=\-\s]+$', arr[index]) is None and len(sarr[index]) >= MIN_NUMBER_OF_WORDS - 2:
            start = get_start_of_word(arr[index])
            if len(start) / len(headers) > 0.5:
                for p in start:
                    col_start[p] += 1
                data_rows.append(index)
                # print(start)

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

    for i in data_rows:
        obj = {
            'data': {},
            'header': {},
            'line_index': i,
            'StructureType': 'table',
            'label_line_index': header_index,
            'label_string': arr[header_index],
        }

        row = [ r.strip() for r in split_row(arr[i], pos) ]

        for j in range(min(len(headers), len(row))):
            obj['header'][str(j)] = headers[j]
            obj['data'][str(j)] = row[j]

        result['table_data'].append(obj)

    r_table[table_name] = result
    return result 


def p5_process_html_table(tables, verbose=True):
    container = {}
    threads = [ threading.Thread(
        target=process_table, 
        args=(table, table_name, container, verbose)
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



def p5_process_html(path, only_extract_html_line=False, verbose=True):
    with open(path, encoding='utf-8') as f:
        soup = bs4.BeautifulSoup(f, features='lxml')
        for t in soup(['script', 'style', 'meta']):
            t.extract()

        pretty_soup_str = soup.prettify()
        normal_soup_str = soup.text

        pretty_soup_str = re.sub('\s+<span', '<span', pretty_soup_str)
        normal_soup_str = re.sub('\s+<span', '<span', normal_soup_str)

    total_tables = 0
    try:
        raw_pretty_tables = pd.read_html(pretty_soup_str)
    except ValueError:
        traceback.print_exc()
        raw_pretty_tables = []
        if verbose:
            print('No pretty table found')

    try:
        raw_normal_tables = pd.read_html(normal_soup_str)
    except ValueError:
        traceback.print_exc()
        raw_normal_tables = []
        if verbose:
            print('No pretty table found')

    raw_tables = None
    if len(raw_pretty_tables) > len(raw_normal_tables):
        raw_tables = raw_pretty_tables
    else:
        raw_tables = raw_normal_tables

    format_html_table(raw_tables)
    tables = separate_tables(raw_tables, table_margin=1)
    r_table = p5_process_html_table(tables, verbose=verbose)
    total_tables = len(tables)


    pretty_soup_str = re.sub('(?<!\>)\n+', '', pretty_soup_str)
    soup = bs4.BeautifulSoup(pretty_soup_str, features='lxml')
    for t in soup(['table']):
        t.extract()

    arr = soup.text.replace('\xa0', ' ').split('\n')
    arr = [ t for t in arr if len(t.strip()) > 0 ]
    sarr = [ re.split('\s{2,}', t) for t in arr ]

    res = [None] * len(arr)

    if verbose: print('requesting url of arr')

    for i in range(0, len(arr), MAX_THREADS):
        threads = [ threading.Thread(
            target=request_header_data_async, 
            args=(arr[i], i, res),
        ) for i in range(i, min(i + MAX_THREADS, len(arr))) ]

        for thread in threads: thread.start()
        for thread in threads: thread.join()

    if verbose: print('finish requesting url of arr')

    if only_extract_html_line:
        p5_process_html_with_lines(
            arr, 
            sarr, 
            res, 
            r_table, 
            table_name='table_' + str(total_tables),
        )

        return r_table

    found_table = False
    for i, line in enumerate(arr):
        if len(line) < 5 or len(sarr[i]) <= 3:
            continue

        if re.search('^[=\-\s]+$', line) is None or len(res[i].keys()) >= 5:
            found_table = True
            break

    if not found_table:
        r_table['table_' + str(total_tables)] = p5_process_html_no_header(
            arr, 
            sarr, 
            res, 
            verbose=verbose,
        )
    else:
        count = total_tables
        i = 0

        found_separator = False
        while i < len(arr):
            if re.search('^[=\-\s]+$', arr[i]): 
                found_separator = True
                start_table = max(0, i - 1)
                i += 1

                while i < len(arr):
                    if re.search('^[=\-\s]+$', arr[i]) or i == len(arr) - 1: 
                        end_table = i

                        p5_process_html_with_header(
                            arr, 
                            sarr, 
                            res, 
                            r_table,
                            start_index=start_table,
                            stop_index=end_table,
                            table_name='table_' + str(count),
                            verbose=verbose,
                        )
                        count += 1

                        if end_table - start_table <= 2: i += 1
                        else: i -= 1
                        break
                    else: i += 1
            else: i += 1

        if not found_separator:
            p5_process_html_with_header(
                arr, 
                sarr, 
                res, 
                r_table, 
                table_name='table_' + str(count),
                verbose=verbose,
            )

    return r_table


def process_table(table, table_name, container, verbose=True):
    table.dropna(how='all', inplace=True, axis=0)
    table.dropna(how='all', inplace=True, axis=1)

    height, width = table.shape
    res = [ None ] * height
    result = {
        'header_data': [],
        'table_data': [],
    }

    if verbose: print('requesting header urls in process_table...')
    for i in range(0, height, MAX_THREADS):
        threads = []
        for i in range(i, min(i + MAX_THREADS, height)):
            items = list(table.iloc[i])
            items = [ item for item in items if isinstance(item, str) ]
            s = ' '.join(items)
            threads.append(threading.Thread(
                target=request_header_data_async,
                args=(s, i, res),
            ))

        for thread in threads: thread.start()
        for thread in threads: thread.join()

    if verbose: print('finish requesting header urls in process_table')

    for i in range(height):
        header_item = {
            'line_index': i,
            'number_of_words': len(res[i].keys()),
            'data': [],
        }

        for header_data_key, header_data_value in res[i].items():
            header_item['data'].append({
                'url': header_data_key,
                'word': header_data_value,
            })

        result['header_data'].append(header_item)


    header_index = 0
    while header_index < height:
        c_nan = table.iloc[header_index].isna().sum()
        if c_nan / width < 0.7:
            break

        header_index += 1

    if header_index >= height: 
        container[table_name] = result 
        return

    columns = table.iloc[header_index].astype(str).tolist()

    # set unknown labels
    c = 0
    for i in range(0, width):
        if pd.isnull(columns[i]) or columns[i] == 'nan':
            columns[i] = 'unknown_' + str(c)
            c += 1

    table.columns = columns
    table.drop(index=table.index[:header_index + 1], inplace=True)


    if verbose: print(table.columns)

    results = table.astype(str).to_dict(orient='index')

    res = list(results.values())
    for obj in res:
        for key, value in obj.items():
            if value is None or value == 'nan':
                obj[key] = ''

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

        result['table_data'].append(new_obj)

    container[table_name] = result


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
        args=(table, table_name, container, verbose)
    ) for table_name, table in tables.items() ]

    # Processing tables...
    if verbose: print('processing tables...')
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if verbose: print('finish processing tables')


    return container


def p5_process_pdf(path, verbose=True):
    MARGIN_Y = 1.0
    A = 100000


    def get_x(obj): return obj['position']['x']
    def get_y(obj): return obj['position']['y']

    with open(path, encoding='utf-8', errors='ignore') as f:
        data = json.load(f)

    # Build histogram
    y = [ get_y(p) for p in data ]
    y.sort()

    cur_y = y[0]
    hist_y = { cur_y: cur_y }

    for yc in y:
        if yc == cur_y: continue
        if yc - cur_y > MARGIN_Y:
            cur_y = yc
        hist_y[yc] = cur_y

    data.sort(key=lambda obj: hist_y[get_y(obj)] * A + get_x(obj))
    slist = {}
    stext = {}
    for obj in data:
        hy = hist_y[get_y(obj)]
        if hy not in slist:
            slist[hy] = []
            stext[hy] = []

        slist[hy].append(obj)
        stext[hy].append(obj['text'])

    sarr = []
    for line, words in stext.items():
        sentence = ''.join(words)
        sarr.append([ word for word in re.split('\s{2,}', sentence) if len(word) > 0 ])

    sres = []
    sheader = [ None ] * len(sarr)
    for line in range(len(sarr)):
        sres.append({
            'is_header': False,
            'data': '',
        })

    def request_row(sarr, line, sres, sheader):
        r, header_data, is_header = request_word(' '.join(sarr[line]))
        sres[line]['is_header'] = is_header
        sres[line]['data'] = r 
        sheader[line] = header_data

    threads = [ threading.Thread(
        target=request_row, 
        args=(sarr, line, sres, sheader)
    ) for line in range(len(sarr)) ]

    if verbose: print('requesting in pdf...')
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if verbose: print('finish requesting in pdf')

    result = {}
    current_table = {
        'header_data': [],
        'table_data': [],
    }
    count = 0

    header_index = 0
    for line, words in enumerate(sarr):
        if sres[line]['is_header']:
            return simplify_json(data)

            # if len(current_table['table_data'] > 0):
            #     result['table_' + str(count)] = current_table 
            #     count += 1

            # current_table = {
            #     'header_data': [],
            #     'table_data': [],
            # }

            # header_index = line

        header_item = {
            'line_index': line,
            'number_of_words': len(sheader[line].keys()),
            'data': [],
        }

        for header_data_key, header_data_value in sheader[line].items():
            header_item['data'].append({
                'url': header_data_key,
                'word': header_data_value,
            })

        current_table['header_data'].append(header_item)

        current_table['table_data'].append({
            'StructureType': 'line',
            'data': {
                '0': ''.join(sarr[line]),
            },
            'header': {
                '0': ''.join(sarr[header_index]),
            },
            'line_index': line,
            'label_line_index': header_index,
            'label_string': sarr[header_index],
        })

    if len(current_table['table_data']) > 0:
        result['table_' + str(count)] = current_table 

    return result

def p5_process_file(path, only_extract_html_line=False, verbose=True):
    ext = path.rsplit('.', 1)[1].lower()

    if ext == 'html':
        return p5_process_html(path, only_extract_html_line=only_extract_html_line, verbose=verbose)
    elif ext == 'xls' or ext == 'xlsx':
        return p5_process_excel(path, verbose=verbose)
    elif ext == 'json':
        return p5_process_pdf(path, verbose=verbose)


if __name__ == '__main__':
    # r = p5_process_html('../data/p5materials/html/c37.html', only_extract_html_line=True, verbose=True)
    r = p5_process_pdf('../data/p5materials/pdf/p5.json')
    # r = p5_process_excel('p5materials/excel/x7.xlsx')
    print(json.dumps(r, indent=2))
