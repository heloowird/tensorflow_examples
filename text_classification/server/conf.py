RESPONSE_STR = '''<html>
                <head>
                    <style type="text/css">
                    </style> 
                </head>
                <body>
                    <div style="margin:20px 5px 20px 20px;">
                    <form action="/poi_classification" method="POST">
                        <label>poi name: </label>
                        <input type="text" name="poi_name" value="%s" style="width:200px">
                        <input type="submit" value="Submit">
                    </form>
                    <div><p>poi type with most prob: %s</p></div>
                    <div><p>prob of top5 poi type: </p></div>
                    <div>%s</div>
                    </div>
                </body></html>'''

MODEL_FILENAME="../lstm_cnn_test/model/model-616000"
FEATURE_FILENAME="../data_v3/poi_name_w2v.voc"
LABEL_FILENAME="../data_v3/class_2_dict_v3"
