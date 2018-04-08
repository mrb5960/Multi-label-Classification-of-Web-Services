import json
from backports import csv
import locale
import codecs

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

parsed_api = {}
def load_unparsed_api_metadata():
    try:
        with open( "output_file.json", "rb") as f:
            apis = json.load(f)
            return apis
    except:
        print
        "Cannot load the unparsed api metadata file!"
        return None

def parse_one_api_metadata(api):
    parsed_api['api_desc'] = api['api_desc']
    parsed_api['api_tags'] = api['api_tags']
    parsed_api['api_name'] = api['api_name']
    # for items in parsed_api['api_desc']:
    #     items.encode('utf8')
    # for items in parsed_api['api_tags']:
    #     items.encode('utf8')
    # for items in parsed_api['api_name']:
    #     items.encode('utf8')
    return parsed_api

def parse_all_apis():
    apis = load_unparsed_api_metadata()
    total_count = len(apis)
    a = "{} movie metadata were loaded!".format(total_count)
    print(a)
    with open("api_data.csv", "w") as f:
        header_was_written = False
        for i, api in enumerate(apis):
            #b = "Processing {} of {}: {}".format(i + 1, total_count, api['api_name'])
            #print(b)
            parsed_api = parse_one_api_metadata(api)
            w = csv.DictWriter(f, parsed_api.keys())
            if not header_was_written:
                w.writeheader()
                header_was_written = True

            try:
                w.writerow(parsed_api)
            except UnicodeEncodeError:
                print(parsed_api)


parse_all_apis()