import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors

def compute_count_matrix(corpus,vocabulary,type='wordxword'):
    count_model = CountVectorizer(vocabulary=vocabulary)
    X = count_model.fit_transform(corpus)
    if type == 'wordxdoc':
        print('*** computing word x document count matrix')
        count_matrix=X.toarray()
    elif type == 'wordxword':
        print('*** computing word x word count matrix')
        Xc = (X.T * X) # this is co-occurrence matrix in sparse csr format
        count_matrix=Xc.toarray() # print out matrix in dense format
    return count_matrix

def return_df(count_matrix,vocabulary):
    nrows=len(vocabulary)
    if count_matrix.shape[0]==nrows:
        df = pd.DataFrame(data=count_matrix)
    elif count_matrix.shape[1]==nrows:
        df = pd.DataFrame(data=np.transpose(count_matrix))
    df = df.set_index([vocabulary])
    return df

def perplexity_over_components(data,range_components):
    perp_components=[]
    for ncomp in range_components:
        lda_model = LatentDirichletAllocation(n_components=ncomp,random_state=122)
        lda_model.fit(data)
        perp = lda_model.perplexity(data)
        perp_components.append(perp)
    return perp_components

def make_perplexity_over_components_plot(var_explained,range_components):
    names= list(map(str, range_components))
    plt.plot(names,var_explained)
    plt.ylabel('Perplexity')
    plt.xlabel('Number of components')
    plt.show()
    
def make_wordcloud_plot(lda_model,terms,stop_words,output_filename):
    cols = [color for name, color in mcolors.XKCD_COLORS.items()]  

    cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=20,
                  color_func=lambda *args, **kwargs: cols[i+20],
                  prefer_horizontal=1.0)

    fig, axes = plt.subplots(4,3, figsize=(20,20), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        comp=lda_model.components_[i,:]
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:20]
        topic_words = dict(sorted_terms)
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=25))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=10, hspace=10)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.show()
    
def get_dominant_topic_df(lda_model,data,corpus):
    lda_output = lda_model.transform(data) # create document x topic matrix
    topicnames = ["Topic" + str(i) for i in range(0, lda_model.n_components,1)]
    docnames = ["Doc" + str(i) for i in range(len(corpus))]
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)  # make pandas dataframe
    dominant_topic = np.argmax(df_document_topic.values, axis=1) # get dominant topic for each document
    df_document_topic['dominant_topic'] = dominant_topic
    return df_document_topic

def make_topic_distribution_plot(df_topic_distribution,output_filename):
    cols = [color for name, color in mcolors.XKCD_COLORS.items()]  
    sizes = list(df_topic_distribution['Num Documents'])
    topicnum = list(map(str,df_topic_distribution['Topic Num']))
    labels=["Topic " + topicnum[i] for i in range(len(topicnum))]
    
    #Relabel after visually inspecting wordclouds (only for ease of inspection)
    labels = ['Talairach (Topic 0)' if x == 'Topic 0' else x for x in labels]
    labels = ['Segmentation (Topic 1)' if x == 'Topic 1' else x for x in labels]
    labels = ['ICBM (Topic 3)' if x == 'Topic 3' else x for x in labels]
    labels = ['SPM (Topic 5)' if x == 'Topic 5' else x for x in labels]
    labels = ['FSL (Topic 9)' if x == 'Topic 9' else x for x in labels]
    labels = ['EEG (Topic 8)' if x == 'Topic 8' else x for x in labels]
    labels = ['Normalization (Topic 2)' if x == 'Topic 2' else x for x in labels]
    labels = ['Smoothing (Topic 10)' if x == 'Topic 10' else x for x in labels]
    
    barpplot_cols=[]
    for i in range(len(labels)):
        barpplot_cols.append(cols[i+20])
    
    # ordering colors
    color_id = [int(x) for x in df_topic_distribution['Topic Num']]
    barpplot_cols_ordered=[]
    for i in range(len(color_id)):
        barpplot_cols_ordered.append(barpplot_cols[color_id[i]])
    
    #generate figure
    fig, ax = plt.subplots(figsize =(16, 9)) 
    ax.barh(labels, sizes, color=barpplot_cols_ordered) 
  
    for s in ['top', 'bottom', 'left', 'right']: # remove axes splines 
        ax.spines[s].set_visible(False) 
  
    ax.xaxis.set_ticks_position('none') # remove ticks 
    ax.yaxis.set_ticks_position('none') 
    ax.axes.set_xlabel('Number of papers',fontsize=12)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.set_tick_params(pad = 5) # add padding between axes and labels 
    ax.yaxis.set_tick_params(pad = 10) 
    ax.grid(b = True, color ='grey', # add x, y gridlines 
        linestyle ='-.', linewidth = 0.5, 
        alpha = 0.5) 
    ax.invert_yaxis()  # show top values  
    
    for i in ax.patches: # add annotation to bars 
        plt.text(i.get_width()+0.5, i.get_y()+0.5,  
                 str(round((i.get_width()), 2)), 
                 fontsize = 10, fontweight ='regular', 
                 color ='grey') 
 
    ax.set_title('Topic distribution across papers (most dominant topic)', # add plot uitle 
             loc ='left',fontsize=15 ) 

    plt.savefig(output_filename, bbox_inches='tight')
    plt.show()