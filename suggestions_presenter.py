import csv
import urllib.request, json
from os import listdir

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def get_parent_taxons(taxon):
    possible_parent_taxons = taxon.get("links", {}).get("parent_taxons", {})
    if any(possible_parent_taxons):
        for parent_taxon in possible_parent_taxons:
            while parent_taxon != {}:
                possible_parent_taxon = parent_taxon.get("links", {}).get("parent_taxons", {})
                if possible_parent_taxon != {}:
                    parent_taxon = possible_parent_taxon[0]
                else:
                    break
            return [parent_taxon['title']]
    else:
        return [""]

def get_taxon_names(content_item, field):
    taxons = content_item.get("links", {}).get(field, {})
    taxon_names = []
    for taxon in taxons:
        parent_taxons = get_parent_taxons(taxon)
        parent_taxons.insert(0, taxon['title'])
        taxon_names.append(parent_taxons)
    return taxon_names

def get_formatted_taxon_names(content_item, field = "taxons", current_taxon_title = None):
    taxon_names = get_taxon_names(content_item, field)
    formatted_names = ""
    if any(taxon_names):
        for taxon in taxon_names:
            taxon.reverse()
            if current_taxon_title is not None:
                taxon.append(current_taxon_title)
            joined_names = " > ".join(taxon)
            joined_names = remove_prefix(joined_names, " > ")
            formatted_names += f"<p class='tab'>{joined_names}</p>"
        return formatted_names
    else:
        if current_taxon_title is not None:
            return current_taxon_title
        else:
            "None"

def get_taxon_name_path(taxon_base_path):
    with urllib.request.urlopen("http://www.gov.uk/api/content/" + taxon_base_path) as url:
        content_item = json.loads(url.read().decode())
        return get_formatted_taxon_names(content_item, 'parent_taxons', content_item["title"])

def get_retagging_content(retagging_info, content_item):
    title = content_item["title"]
    description = content_item["description"]
    body = content_item.get("details", {}).get("body", "").replace('\n', '<br/>')
    current_taxons = get_formatted_taxon_names(content_item)
    full_url = "https://www.gov.uk" + content_item["base_path"]
    move_from = get_taxon_name_path(retagging_info["current_taxon_base_path"])
    move_to = get_taxon_name_path(retagging_info["suggestion_base_path"])
    move_to_other_content_url = f"https://www.gov.uk{retagging_info['suggestion_base_path']}"
    return f"<h2 class='inline'>{title}</h2><a class='inline' href='{full_url}' target='_blank'>Link to page</a><h4>Move from: {move_from}</h4><h4>Move to: {move_to}</h4><a class='tab' target='_blank' href='{move_to_other_content_url}'>Other content tagged to it</a><h4>Currently tagged to</h4>{current_taxons}<h4>Description</h4><p>{description}</p><h4>Body</h4><p>{body}</p>"

def get_untagging_content(untagging_info, content_item):
    title = content_item["title"]
    description = content_item["description"]
    body = content_item.get("details", {}).get("body", "").replace('\n', '<br/>')
    current_taxons = get_formatted_taxon_names(content_item)
    full_url = "https://www.gov.uk" + content_item["base_path"]
    move_from = get_taxon_name_path(untagging_info["current_taxon_base_path"])
    return f"<h2 class='inline'>{title}</h2><a class='inline' href='{full_url}' target='_blank'>Link to page</a><h4>Tag to remove: {move_from}</h4><h4>Currently tagged to</h4>{current_taxons}<h4>Description</h4><p>{description}</p><h4>Body</h4><p>{body}</p>"

dir_name = "for_review"
output_filenames = []
for filename in listdir(dir_name):
    print("Presenting: " + filename)
    filename_to_open = dir_name + "/" + filename
    if "retag" in filename_to_open:
        output = []
        already_appended = []
        with open(filename_to_open, 'r') as csvfile:
            content_to_retag = csv.DictReader(csvfile)
            for retagging_info in content_to_retag:
                retag_output = {}
                with urllib.request.urlopen("http://www.gov.uk/api/content/" + retagging_info["content_to_retag_base_path"]) as url:
                    content_item = json.loads(url.read().decode())
                    retag_output["id"] = retagging_info["content_to_retag_base_path"]
                    retag_output["question"] = "Should this be retagged from its current taxon to this suggested one?"
                    retag_output["content"] = get_retagging_content(retagging_info, content_item)
                    retag_output["url"] = ""
                    unique_name = retagging_info["content_to_retag_base_path"] + retagging_info["current_taxon_base_path"]
                    if unique_name not in already_appended:
                        already_appended.append(unique_name)
                        output.append(retag_output)
        if any(output):
            keys = output[0].keys()
            output_filename = filename.split(".csv")[0] + "_items.csv"
            output_filenames.append(output_filename)
            with open(output_filename, 'w') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(output)
    if "untag" in filename_to_open:
        output = []
        already_appended = []
        with open('depth_first_content_to_untag_Money.csv', 'r') as csvfile:
            content_to_untag = csv.DictReader(csvfile)
            for untagging_info in content_to_untag:
                retag_output = {}
                with urllib.request.urlopen("http://www.gov.uk/api/content/" + untagging_info["content_to_retag_base_path"]) as url:
                    content_item = json.loads(url.read().decode())
                    retag_output["id"] = untagging_info["content_to_retag_base_path"]
                    retag_output["question"] = "Should this page be untagged?"
                    retag_output["content"] = get_untagging_content(untagging_info, content_item)
                    retag_output["url"] = ""
                    unique_name = untagging_info["content_to_retag_base_path"] + untagging_info["current_taxon_base_path"]
                    if unique_name not in already_appended:
                        already_appended.append(unique_name)
                        output.append(retag_output)
        if any(output):
            keys = output[0].keys()
            output_filename = filename.split(".csv")[0] + "_items.csv"
            output_filenames.append(output_filename)
            with open(output_filename, 'w') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(output)
for output_filename in output_filenames:
    print(output_filename)