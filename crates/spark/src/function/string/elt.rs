// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use datafusion_macros::user_doc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, PrimitiveArray};
use arrow::compute::{can_cast_types, cast, interleave, nullif};
use arrow::datatypes::DataType::{Binary, BinaryView, Int64, LargeBinary, Utf8};
use arrow::datatypes::{DataType, Int64Type};
use datafusion_common::{DataFusionError, Result, exec_err, plan_datafusion_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_functions::utils::make_scalar_function;

/// Spark-compatible `elt` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#elt>
#[user_doc(
    doc_section(label = "String Functions"),
    description = "Returns the `n`-th of the following arguments (1-based). \
        The result is a string, or binary when every value argument is binary. \
        DataFusion has no built-in `elt`. \
        A NULL `n` returns NULL, as does picking a NULL argument. \
        If `n` is less than 1 or greater than the number of arguments, the result is NULL, \
        or an error when `datafusion.execution.enable_ansi_mode` is true.",
    syntax_example = "elt(n, expr1[, ..., exprN])",
    sql_example = r#"```sql
> SELECT elt(1, 'scala', 'java');
scala
> SELECT elt(2, 'a', 1);
1
> SELECT elt(3, 'scala', 'java');
NULL
```"#,
    argument(
        name = "n",
        description = "Integer index, 1-based, cast to BIGINT. NULL returns NULL."
    ),
    argument(
        name = "expr",
        description = "At least one value to choose from. If every value is binary the result is \
            binary, otherwise each is cast to a string."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkElt {
    signature: Signature,
}

impl Default for SparkElt {
    fn default() -> Self {
        SparkElt::new()
    }
}

impl SparkElt {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkElt {
    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn name(&self) -> &str {
        "elt"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        // `arg_types` are already coerced: the index is `Int64` and the value
        // arguments are either all binary or all `Utf8`. Spark's `Elt.dataType`
        // is the type of the first value argument, so keep binary when every
        // value argument is binary and fall back to a string otherwise.
        let value_types = arg_types.get(1..).unwrap_or(&[]);
        if all_binary(value_types) {
            Ok(widest_binary_type(value_types))
        } else {
            Ok(Utf8)
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let enable_ansi_mode = args.config_options.execution.enable_ansi_mode;
        make_scalar_function(
            move |arrays: &[ArrayRef]| elt(arrays, enable_ansi_mode),
            vec![],
        )(&args.args)
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let length = arg_types.len();
        if length < 2 {
            return Err(plan_datafusion_err!(
                "ELT function expects at least 2 arguments: index, value1"
            ));
        }

        let idx_dt: &DataType = &arg_types[0];
        if *idx_dt != Int64 && !can_cast_types(idx_dt, &Int64) {
            return Err(DataFusionError::Plan(format!(
                "ELT index must be Int64 (or castable to Int64), got {idx_dt:?}"
            )));
        }

        // Spark (`EltTypeCoercion`) keeps BINARY output only when every value
        // argument is binary. A NULL-typed argument is not binary, so it forces
        // the string path, matching `children.tail.forall(_ == BinaryType)`.
        let value_types = &arg_types[1..];
        let mut coerced = Vec::with_capacity(length);
        coerced.push(Int64);
        if all_binary(value_types) {
            let target = widest_binary_type(value_types);
            coerced.resize(length, target);
        } else {
            coerced.resize(length, Utf8);
        }
        Ok(coerced)
    }
}

/// True when `types` is non-empty and every type is a binary type. Mirrors
/// Spark treating only `BinaryType` as binary: a NULL-typed (`DataType::Null`)
/// argument is not binary and forces the string path.
fn all_binary(types: &[DataType]) -> bool {
    !types.is_empty() && types.iter().all(|dt| dt.is_binary())
}

/// Widest binary type among `types`: `BinaryView`, then `LargeBinary`, then
/// `Binary` (which `FixedSizeBinary` is cast to).
fn widest_binary_type(types: &[DataType]) -> DataType {
    if types.contains(&BinaryView) {
        BinaryView
    } else if types.contains(&LargeBinary) {
        LargeBinary
    } else {
        Binary
    }
}

fn elt(args: &[ArrayRef], enable_ansi_mode: bool) -> Result<ArrayRef> {
    let n_rows = args[0].len();

    let idx: &PrimitiveArray<Int64Type> =
        args[0].as_primitive_opt::<Int64Type>().ok_or_else(|| {
            DataFusionError::Plan(format!(
                "ELT function: first argument must be Int64 (got {:?})",
                args[0].data_type()
            ))
        })?;

    // Planning already coerces the values to one type, but cast here too so the
    // kernel also works on uncoerced input. `interleave` then gathers the picked
    // values and keeps their type, so binary input stays binary.
    let value_types: Vec<DataType> =
        args.iter().skip(1).map(|a| a.data_type().clone()).collect();
    let target = if all_binary(&value_types) {
        widest_binary_type(&value_types)
    } else {
        Utf8
    };
    let casted = args
        .iter()
        .skip(1)
        .map(|a| cast(a, &target))
        .collect::<Result<Vec<_>, _>>()?;
    let values: Vec<&dyn Array> = casted.iter().map(|a| a.as_ref()).collect();
    let num_values = values.len();

    // For each row, the (column, row) pair to pull from, plus a mask marking the
    // rows that must be NULL (NULL index or out-of-range index). Those rows point
    // at an arbitrary valid slot and are nulled out afterwards with `nullif`.
    let mut indices: Vec<(usize, usize)> = Vec::with_capacity(n_rows);
    let mut force_null: Vec<bool> = Vec::with_capacity(n_rows);

    for i in 0..n_rows {
        if idx.is_null(i) {
            indices.push((0, i));
            force_null.push(true);
            continue;
        }

        let index = idx.value(i);

        if index < 1 || (index as usize) > num_values {
            if enable_ansi_mode {
                return exec_err!(
                    "The index {index} is out of bounds. The array has {num_values} elements."
                );
            }
            indices.push((0, i));
            force_null.push(true);
            continue;
        }

        indices.push(((index as usize) - 1, i));
        force_null.push(false);
    }

    let selected = interleave(&values, &indices)?;
    let mask = BooleanArray::from(force_null);
    Ok(nullif(&selected, &mask)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, Int64Array, LargeBinaryArray, StringArray};
    use datafusion_common::cast::{as_binary_array, as_string_array};
    use std::sync::Arc;

    fn run_elt_arrays(arrs: Vec<ArrayRef>) -> Result<StringArray> {
        run_elt_arrays_with(arrs, false)
    }

    fn run_elt_arrays_with(arrs: Vec<ArrayRef>, ansi: bool) -> Result<StringArray> {
        let arr = elt(&arrs, ansi)?;
        Ok(as_string_array(&arr)?.clone())
    }

    #[test]
    fn elt_utf8_basic() -> Result<()> {
        let idx = Arc::new(Int64Array::from(vec![
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(0),
            None,
        ]));
        let v1 = Arc::new(StringArray::from(vec![
            Some("a1"),
            Some("a2"),
            Some("a3"),
            Some("a4"),
            Some("a5"),
            Some("a6"),
        ]));
        let v2 = Arc::new(StringArray::from(vec![
            Some("b1"),
            Some("b2"),
            None,
            Some("b4"),
            Some("b5"),
            Some("b6"),
        ]));
        let v3 = Arc::new(StringArray::from(vec![
            Some("c1"),
            Some("c2"),
            Some("c3"),
            None,
            Some("c5"),
            Some("c6"),
        ]));

        let out = run_elt_arrays(vec![idx, v1, v2, v3])?;
        assert_eq!(out.len(), 6);
        assert_eq!(out.value(0), "a1");
        assert_eq!(out.value(1), "b2");
        assert_eq!(out.value(2), "c3");
        assert!(out.is_null(3));
        assert!(out.is_null(4));
        assert!(out.is_null(5));
        Ok(())
    }

    #[test]
    fn elt_int64_basic() -> Result<()> {
        let idx = Arc::new(Int64Array::from(vec![Some(2), Some(1), Some(2)]));
        let v1 = Arc::new(Int64Array::from(vec![Some(10), Some(20), Some(30)]));
        let v2 = Arc::new(Int64Array::from(vec![Some(100), None, Some(300)]));

        let out = run_elt_arrays(vec![idx, v1, v2])?;
        assert_eq!(out.len(), 3);
        assert_eq!(out.value(0), "100");
        assert_eq!(out.value(1), "20");
        assert_eq!(out.value(2), "300");
        Ok(())
    }

    #[test]
    fn elt_out_of_range_all_null() -> Result<()> {
        let idx = Arc::new(Int64Array::from(vec![Some(5), Some(-1), Some(0)]));
        let v1 = Arc::new(StringArray::from(vec![Some("x"), Some("y"), Some("z")]));
        let v2 = Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")]));

        let out = run_elt_arrays(vec![idx, v1, v2])?;
        assert!(out.is_null(0));
        assert!(out.is_null(1));
        assert!(out.is_null(2));
        Ok(())
    }

    #[test]
    fn elt_utf8_returns_utf8() -> Result<()> {
        let idx = Arc::new(Int64Array::from(vec![Some(1)]));
        let v1 = Arc::new(StringArray::from(vec![Some("scala")]));
        let v2 = Arc::new(StringArray::from(vec![Some("java")]));

        let out = run_elt_arrays(vec![idx, v1, v2])?;
        assert_eq!(out.data_type(), &Utf8);
        Ok(())
    }

    #[test]
    fn elt_all_binary_returns_binary() -> Result<()> {
        let idx = Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(3)]));
        let v1 = Arc::new(BinaryArray::from_opt_vec(vec![
            Some(&b"a1"[..]),
            Some(&b"a2"[..]),
            Some(&b"a3"[..]),
        ]));
        let v2 = Arc::new(BinaryArray::from_opt_vec(vec![
            Some(&b"b1"[..]),
            None,
            Some(&b"b3"[..]),
        ]));

        let arr = elt(&[idx, v1, v2], false)?;
        assert_eq!(arr.data_type(), &Binary);
        let out = as_binary_array(&arr)?;
        assert_eq!(out.value(0), b"a1");
        // Picks the NULL value at index 2, row 1.
        assert!(out.is_null(1));
        assert!(out.is_null(2)); // index 3 is out of range -> NULL
        Ok(())
    }

    #[test]
    fn elt_return_type_all_binary() -> Result<()> {
        let udf = SparkElt::new();
        assert_eq!(udf.return_type(&[Int64, Binary, Binary])?, Binary);
        // The widest binary type wins.
        assert_eq!(udf.return_type(&[Int64, Binary, LargeBinary])?, LargeBinary);
        assert_eq!(udf.return_type(&[Int64, Binary, BinaryView])?, BinaryView);
        Ok(())
    }

    #[test]
    fn elt_return_type_falls_back_to_utf8() -> Result<()> {
        let udf = SparkElt::new();
        // A mix of binary and non-binary value arguments -> string.
        assert_eq!(udf.return_type(&[Int64, Binary, Utf8])?, Utf8);
        // A NULL-typed argument is not binary -> string.
        assert_eq!(udf.return_type(&[Int64, Binary, DataType::Null])?, Utf8);
        Ok(())
    }

    #[test]
    fn elt_coerce_types_all_binary_stays_binary() -> Result<()> {
        let udf = SparkElt::new();
        let coerced = udf.coerce_types(&[Int64, Binary, LargeBinary])?;
        // Index -> Int64, value args -> widest binary type.
        assert_eq!(coerced, vec![Int64, LargeBinary, LargeBinary]);

        // A NULL-typed value argument forces the string path.
        let coerced = udf.coerce_types(&[Int64, Binary, DataType::Null])?;
        assert_eq!(coerced, vec![Int64, Utf8, Utf8]);
        Ok(())
    }

    #[test]
    fn elt_largebinary_returns_largebinary() -> Result<()> {
        let idx = Arc::new(Int64Array::from(vec![Some(2), Some(1)]));
        let v1 = Arc::new(LargeBinaryArray::from_opt_vec(vec![
            Some(&b"a1"[..]),
            Some(&b"a2"[..]),
        ]));
        let v2 = Arc::new(LargeBinaryArray::from_opt_vec(vec![
            Some(&b"b1"[..]),
            Some(&b"b2"[..]),
        ]));

        let arr = elt(&[idx, v1, v2], false)?;
        assert_eq!(arr.data_type(), &LargeBinary);
        Ok(())
    }
}
