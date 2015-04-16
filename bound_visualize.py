def bound_visualize(clf,data,title):
    ### 決定境界の可視化

    fig=plt.figure()
    # Parameters for plot
    n_classes = 2
    plot_colors = "br"
    plot_step = 0.005

    #グラフ描画時の説明変数 x、yの最大値＆最小値を算出。
    #グラフ描画のメッシュを定義
    x_min = 1.5
    x_max = 5.0
    y_min = 4
    y_max = 8.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    #各メッシュ上での分類を計算
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    #分類を等高線フィールドプロットでプロット
    cs = plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

    plt.scatter(data[u'Sepal.Width'][data[u'Species']=='setosa']
            ,data[u'Sepal.Length'][data[u'Species']=='setosa']
            ,c='b',marker='o',label='setosa')
    plt.scatter(data[u'Sepal.Width'][data[u'Species']=='virginica']
            ,data[u'Sepal.Length'][data[u'Species']=='virginica']
            ,c='r',marker='x',label='virginica')
    plt.scatter(data[u'Sepal.Width'][data[u'Species']=='versicolor']
            ,data[u'Sepal.Length'][data[u'Species']=='versicolor']
            ,c='g',marker='^',label='versicolor')
    plt.title(title)
    plt.xlabel('Sepal.Width')
    plt.ylabel('Sepal.Length')
    plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0)

    plt.show()
