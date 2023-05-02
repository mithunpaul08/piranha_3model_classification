#given a dictionary and a key, increase its count if it exists, else add it
def increase_counter(key,dict_to_check):
    if key in dict_to_check:
        old_value=dict_to_check[key]
        dict_to_check[key]=old_value+1
    else:
        dict_to_check[key]=1
    return dict_to_check

def given_relative_aggreement_calculate_cohen(p0,pe):
    #p0- relative observed agreement among raters
    #pe-hypothetical probability of chance agreement which in this case is 50%
    k=(p0-pe)/(1-pe)
    return k

if __name__=="__main__":
    print(given_relative_aggreement_calculate_cohen(1,0.25))