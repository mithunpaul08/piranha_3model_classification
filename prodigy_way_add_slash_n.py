
import json
YOUR_LONG_TEXT="Hello Jesse,\nMy name is Sarah and I work for facebook as a marketing manager at the menlo park location. Would you like to start an absolutely real work at home job with a salary of 2500 USD monthly and without investment or startup charges. If yes click here to submit your resume. Alternatively you can give me a call to my personal cell phone number: (423) -234- 9554. \n\n\n\nRegards, \nSarah Adamik\nMarketing Manager\nFacebook Inc,\n1, Hacker Way, Menlo Park, CA 94025\n713.853.4764 Office\n713-853-9828 Fax\nmailto:sara.adamik@facebook.com\ninstagram: @saraha\nwww.facebook.com\n\n\n\n\n"
contracts = YOUR_LONG_TEXT.split('\n')
# you might also want to do some stripping of whitespace etc. here

for contract in contracts:
    task = {'text': contract}
    print(json.dumps(task))

######

