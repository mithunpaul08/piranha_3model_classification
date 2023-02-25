#given a dictionary and a key, increase its count if it exists, else add it
def increase_counter(key,dict_to_check):
    if key in dict_to_check:
        old_value=dict_to_check[key]
        dict_to_check[key]=old_value+1
    else:
        dict_to_check[key]=1
    return dict_to_check