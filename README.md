# 2023 Fall 分布式机器学习lab1 Warmup

## Todo List
- [x] CUDA编程实现LayerNorm定制化算子的前向传播
- [x] CUDA编程实现LayerNorm定制化算子的反向传播
- [x] 和pytorch.nn.LayerNorm正确性和性能对比
- [x] 使用pytorch profiler分析算子性能
- [x] 完成实验报告
- [] 参考[Triton](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)进行优化
  
## 定制化LayerNorm算子

### 编译和安装

软件环境：pytorch2.1.0 + CUDA12.1

```shell
cd layerNorm_cuda_extension
python setup install --user
```

编译过程中，如遇到报错：

```bash
/usr/include/pybind11/detail/../cast.h: In function ‘typename pybind11::detail::type_caster<typename pybind11::detail::intrinsic_type<T>::type
>::cast_op_type<T> pybind11::detail::cast_op(make_caster<T>&)’:
/usr/include/pybind11/detail/../cast.h:45:120: error: expected template-name before ‘<’ token
   45 |     return caster.operator typename make_caster<T>::template cast_op_type<T>();
      |                                                                                                                        ^
/usr/include/pybind11/detail/../cast.h:45:120: error: expected identifier before ‘<’ token
/usr/include/pybind11/detail/../cast.h:45:123: error: expected primary-expression before ‘>’ token
   45 |     return caster.operator typename make_caster<T>::template cast_op_type<T>();
      |                                                                                                                           ^
/usr/include/pybind11/detail/../cast.h:45:126: error: expected primary-expression before ‘)’ token
   45 |     return caster.operator typename make_caster<T>::template cast_op_type<T>();
```

解决方案：在`/usr/include/pybind11/cast.h`中进行以下修改：

```cpp
-    return caster.operator typename make_caster<T>::template cast_op_type<T>();
+    return caster;
```

### 正确性和性能测试

```shell
cd ..
python custom_layerNorm.py
```

如果没有assert则说明在误差范围内正确，并打印前向和反向的运行时间比较。

执行

```shell
python custom_layerNorm.py --profile
```

将打印pytorch profile的结果，并将完整结果以json格式存储在profile文件夹中，可以通过chrome浏览器chrome://tracing/进行可视化。