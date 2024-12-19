python project/parallel_check.py
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/thomasonzhou/src/minitorch-module-3-thomasonzhou/minitorch/fast_ops.py 
(181)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/thomasonzhou/src/minitorch-module-3-thomasonzhou/minitorch/fast_ops.py (181) 
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                | 
        out: Storage,                                                        | 
        out_shape: Shape,                                                    | 
        out_strides: Strides,                                                | 
        in_storage: Storage,                                                 | 
        in_shape: Shape,                                                     | 
        in_strides: Strides,                                                 | 
    ) -> None:                                                               | 
                                                                             | 
        if _stride_aligned(out_shape, out_strides, in_shape, in_strides):    | 
            for i in prange(len(out)):---------------------------------------| #0
                out[i] = fn(in_storage[i])                                   | 
        else:                                                                | 
            out_idx = np.empty_like(out_shape, dtype=np.int64)               | 
            in_idx = np.empty_like(in_shape, dtype=np.int64)                 | 
            for i in prange(len(out)):---------------------------------------| #1
                to_index(i, out_shape, out_idx)                              | 
                broadcast_index(out_idx, out_shape, in_shape, in_idx)        | 
                                                                             | 
                out_pos = index_to_position(out_idx, out_strides)            | 
                in_pos = index_to_position(in_idx, in_strides)               | 
                out[out_pos] = fn(in_storage[in_pos])                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/thomasonzhou/src/minitorch-module-3-thomasonzhou/minitorch/fast_ops.py 
(230)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/thomasonzhou/src/minitorch-module-3-thomasonzhou/minitorch/fast_ops.py (230) 
--------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                   | 
        out: Storage,                                                           | 
        out_shape: Shape,                                                       | 
        out_strides: Strides,                                                   | 
        a_storage: Storage,                                                     | 
        a_shape: Shape,                                                         | 
        a_strides: Strides,                                                     | 
        b_storage: Storage,                                                     | 
        b_shape: Shape,                                                         | 
        b_strides: Strides,                                                     | 
    ) -> None:                                                                  | 
        if _stride_aligned(a_shape, a_strides, b_shape, b_strides) \            | 
            and _stride_aligned(a_shape, a_strides, out_shape, out_strides):    | 
            for i in prange(len(out)):------------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                         | 
        else:                                                                   | 
            out_idx = np.empty_like(out_shape, dtype=np.int64)                  | 
            a_idx = np.empty_like(a_shape, dtype=np.int64)                      | 
            b_idx = np.empty_like(b_shape, dtype=np.int64)                      | 
            for i in prange(len(out)):------------------------------------------| #3
                to_index(i, out_shape, out_idx)                                 | 
                broadcast_index(out_idx, out_shape, a_shape, a_idx)             | 
                broadcast_index(out_idx, out_shape, b_shape, b_idx)             | 
                                                                                | 
                out_pos = index_to_position(out_idx, out_strides)               | 
                a_pos = index_to_position(a_idx, a_strides)                     | 
                b_pos = index_to_position(b_idx, b_strides)                     | 
                                                                                | 
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])           | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/thomasonzhou/src/minitorch-module-3-thomasonzhou/minitorch/fast_ops.py 
(284)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/thomasonzhou/src/minitorch-module-3-thomasonzhou/minitorch/fast_ops.py (284) 
-------------------------------------------------------------------|loop #ID
    def _reduce(                                                   | 
        out: Storage,                                              | 
        out_shape: Shape,                                          | 
        out_strides: Strides,                                      | 
        a_storage: Storage,                                        | 
        a_shape: Shape,                                            | 
        a_strides: Strides,                                        | 
        reduce_dim: int,                                           | 
    ) -> None:                                                     | 
        a_idx = np.empty_like(a_shape, dtype=np.int64)             | 
        out_idx = np.empty_like(out_shape, dtype=np.int64)         | 
        for i in prange(len(a_storage)):---------------------------| #4
            to_index(i, a_shape, a_idx)                            | 
            broadcast_index(a_idx, a_shape, out_shape, out_idx)    | 
                                                                   | 
            a_pos = index_to_position(a_idx, a_strides)            | 
            out_pos = index_to_position(out_idx, out_strides)      | 
                                                                   | 
            out[out_pos] = fn(out[out_pos], a_storage[a_pos])      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/thomasonzhou/src/minitorch-module-3-thomasonzhou/minitorch/fast_ops.py 
(308)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/thomasonzhou/src/minitorch-module-3-thomasonzhou/minitorch/fast_ops.py (308) 
----------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                  | 
    out: Storage,                                                                             | 
    out_shape: Shape,                                                                         | 
    out_strides: Strides,                                                                     | 
    a_storage: Storage,                                                                       | 
    a_shape: Shape,                                                                           | 
    a_strides: Strides,                                                                       | 
    b_storage: Storage,                                                                       | 
    b_shape: Shape,                                                                           | 
    b_strides: Strides,                                                                       | 
) -> None:                                                                                    | 
    """NUMBA tensor matrix multiply function.                                                 | 
                                                                                              | 
    Should work for any tensor shapes that broadcast as long as                               | 
                                                                                              | 
    ```                                                                                       | 
    assert a_shape[-1] == b_shape[-2]                                                         | 
    ```                                                                                       | 
                                                                                              | 
    Optimizations:                                                                            | 
                                                                                              | 
    * Outer loop in parallel                                                                  | 
    * No index buffers or function calls                                                      | 
    * Inner loop should have no global writes, 1 multiply.                                    | 
                                                                                              | 
                                                                                              | 
    Args:                                                                                     | 
    ----                                                                                      | 
        out (Storage): storage for `out` tensor                                               | 
        out_shape (Shape): shape for `out` tensor                                             | 
        out_strides (Strides): strides for `out` tensor                                       | 
        a_storage (Storage): storage for `a` tensor                                           | 
        a_shape (Shape): shape for `a` tensor                                                 | 
        a_strides (Strides): strides for `a` tensor                                           | 
        b_storage (Storage): storage for `b` tensor                                           | 
        b_shape (Shape): shape for `b` tensor                                                 | 
        b_strides (Strides): strides for `b` tensor                                           | 
                                                                                              | 
    Returns:                                                                                  | 
    -------                                                                                   | 
        None : Fills in `out`                                                                 | 
                                                                                              | 
    """                                                                                       | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                    | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                    | 
                                                                                              | 
    for batch in prange(out_shape[0]):--------------------------------------------------------| #8
        for i in prange(a_shape[-2]):---------------------------------------------------------| #7
            for k in prange(b_shape[-1]):-----------------------------------------------------| #6
                out_pos = batch * out_strides[0] + i * out_strides[1] + k * out_strides[2]    | 
                total = 0                                                                     | 
                for j in prange(a_shape[-1]): # common dim------------------------------------| #5
                    a_pos = batch * a_batch_stride + i * a_strides[1] + j * a_strides[2]      | 
                    b_pos = batch * b_batch_stride + j * b_strides[1] + k * b_strides[2]      | 
                    total += a_storage[a_pos] * b_storage[b_pos]                              | 
                out[out_pos] = total                                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #8, #7).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--7 --> rewritten as a serial loop
      +--6 --> rewritten as a serial loop
         +--5 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--7 (parallel)
      +--6 (parallel)
         +--5 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--7 (serial)
      +--6 (serial)
         +--5 (serial)


 
Parallel region 0 (loop #8) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
