import artm

def topic_model(class_ids, dictionary, num_of_topics, num_back, tau, tf):

    names_of_topics = [str(x) for x in range(num_of_topics)]    
    dictionary.filter(min_tf=tf, class_id='subjects')
    dictionary.filter(min_tf=tf, class_id='objects')
    dictionary.filter(min_tf=tf, class_id='pairs')

    model = artm.ARTM(num_topics=num_of_topics,
                      #reuse_theta=True,
                      cache_theta=True,
                      topic_names=names_of_topics,
                      class_ids=class_ids, 
                      #regularizers=regularizers_artm,
                      dictionary=dictionary)

    model.scores.add(artm.PerplexityScore(name='PerplexityScore',
                                      dictionary=dictionary))
    
    model.scores.add(artm.SparsityPhiScore(name = 'SparcityPhiScore',
                                           topic_names=model.topic_names[:-num_back]))

    model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhiRegularizer',
                                                            class_ids=class_ids,
                                                            topic_names=model.topic_names[:-num_back],tau = -tau))
    model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SmoothPhiRegularizer',
                                                            class_ids=class_ids,
                                                            topic_names=model.topic_names[-num_back:],tau = tau))


    model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorRegularizer',
                                                          class_ids=class_ids,
                                                          topic_names=model.topic_names[:-num_back], tau=tau))
    model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SparseThetaRegularizer',
                                                            topic_names=model.topic_names[-num_back], tau = tau))
    return model


def so_balance(batch_vectorizer, dictionary):
    scores = []
    best = 0
    for x in tqdm(np.arange(0.05,1,0.05)):
        class_ids = {
                    'subjects': x,
                    'objects': (1-x),
                    'pairs': 0,
                    'neg_pol': 0,
                    'pos_pol': 0
                } 
        model = topic_model(class_ids,dictionary, 3,1,2,2)
        model.fit_offline(batch_vectorizer, num_collection_passes=40)

        theta = model.get_theta()
        X = theta.as_matrix()[:-1].T

        y_pred = X.argmax(axis=1)
        pr,rec, f1 = precision_score(y_true,y_pred), recall_score(y_true,y_pred), f1_score(y_true,y_pred)
        scores.append({"score": f1, "class_ids":class_ids})
        if f1>best:
            best = f1
            best_p = class_ids
    return best, best_p, scores