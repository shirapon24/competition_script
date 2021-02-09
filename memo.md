# こちらはコードもメモです。

## pandas
### query
 - 商品名に「つまり」が含まれている行を抜き出す。
    - _df.query('商品名.str.contains("つまみ") and 商品名 == 商品名')