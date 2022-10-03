from flask import Flask, request, json
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# sentence embeddings
import tensorflow as tf
import tensorflow_hub as hub
import os
import psycopg2


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)
# load numpy file
name_embdng_dict = np.load('name_embdng.npy', allow_pickle='TRUE').item()
ocr_text_embdng_dict = np.load('ocr_text_embdng.npy', allow_pickle='TRUE').item()

load_dotenv()

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))



def score(query_vec,Name,OCR_text):
    similarity_score = (0.5 * cosine(query_vec,Name))+ (0.5 * cosine(query_vec,OCR_text))
    return similarity_score

# load template data from Postgres DB
conn = psycopg2.connect(host = os.getenv('host'), port = os.getenv('port'), database = os.getenv('database'), user = os.getenv('user'), password = os.getenv('password'))
cur = conn.cursor()
cur.execute("""SELECT "frontImageUrl", "url", "name", "templateId_new" from all_templates""")
tuples = cur.fetchall()
df_all_tmplt = pd.DataFrame(tuples, columns = ['frontImageUrl', 'url', 'name', 'templateId_new'])
cur.close()
conn.close()


app = Flask(__name__)


@app.route('/template_search', methods=['POST'])
def query():

    # query embedding
    request_data = request.get_json()
    query_vec = model([request_data["enter_query"]])[0]

    # calculate similarity score for each template
    templt_info_list = []
    for nm_i,ocr_i in zip(name_embdng_dict, ocr_text_embdng_dict):
        try:
            sim = score(query_vec, name_embdng_dict[nm_i], ocr_text_embdng_dict[ocr_i])
            tmplt_info = [nm_i, ocr_i, sim]
            templt_info_list.append(tmplt_info)
        except Exception as e:
            print(e)

    # convert templt_info_list into dataframe & sort by similarity score
    df = pd.DataFrame(templt_info_list, columns=['tmplt_id_1', 'tmplt_id_2', 'Sim Score'])
    df_top = df.sort_values(by=['Sim Score'], ascending=False).iloc[0:10]

    # match top 10 templateid in database containing all templates
    df_matching_results = df_top.merge(df_all_tmplt, left_on='tmplt_id_1', right_on='templateId_new')[['name', 'frontImageUrl', 'url', 'templateId_new', 'tmplt_id_1', 'Sim Score']]
    
    # save bucket links for images
    bucketName = 'video-generation/'
    folderName = 'template-images/'
    df_matching_results['bucket_link'] = df_matching_results['url'].apply(lambda url : bucketName + folderName + url.split('/')[-2] + '.jpg')
    
    
    # return the matching results in json format
    output = df_matching_results[['name','frontImageUrl','bucket_link']].to_dict('record')
    response = json.dumps(output, indent=2)
    return response


#app.run(port=8000,debug = False)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8052,debug = False)
    
    
    
# stuff added in new-branch (branch)
