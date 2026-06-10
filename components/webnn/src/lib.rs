/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use ort::session::Session;
use ort::value::Value;
use rustnn::converters::OnnxConverter;
use rustnn::{ConstantData, GraphConverter, GraphInfo, Operand, OperandDescriptor,
             OperandKind, Operation, DataType as RnnDtype};

pub use rustnn::graph::Dimension;

pub struct GraphNode {
    pub op: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub data_type: u32,
    pub shape: Vec<u32>,
    pub attrs: HashMap<String, f64>,
    pub data: Option<Vec<u8>>,
}

pub struct RunResult {
    pub outputs: Vec<Vec<u8>>,
}

fn to_dtype(v: u32) -> RnnDtype {
    match v { 0 => RnnDtype::Float32, 1 => RnnDtype::Float16, 2 => RnnDtype::Int32, 3 => RnnDtype::Uint32,
        4 => RnnDtype::Int64, 5 => RnnDtype::Uint64, 6 => RnnDtype::Int8, 7 => RnnDtype::Uint8, _ => RnnDtype::Float32 }
}

fn make_op(op: &str, ids: &[u32], out: u32) -> Result<Operation, String> {
    Ok(match op {
        "add" => Operation::Add { a: ids[0], b: ids[1], options: None, outputs: vec![out] },
        "sub" => Operation::Sub { a: ids[0], b: ids[1], options: None, outputs: vec![out] },
        "mul" => Operation::Mul { a: ids[0], b: ids[1], options: None, outputs: vec![out] },
        "div" => Operation::Div { a: ids[0], b: ids[1], options: None, outputs: vec![out] },
        "pow" => Operation::Pow { a: ids[0], b: ids[1], options: None, outputs: vec![out] },
        "relu" => Operation::Relu { input: ids[0], options: None, outputs: vec![out] },
        "sigmoid" => Operation::Sigmoid { input: ids[0], options: None, outputs: vec![out] },
        "tanh" => Operation::Tanh { input: ids[0], options: None, outputs: vec![out] },
        "abs" => Operation::Abs { input: ids[0], options: None, outputs: vec![out] },
        "neg" => Operation::Neg { input: ids[0], options: None, outputs: vec![out] },
        "exp" => Operation::Exp { input: ids[0], options: None, outputs: vec![out] },
        "log" => Operation::Log { input: ids[0], options: None, outputs: vec![out] },
        "sqrt" => Operation::Sqrt { input: ids[0], options: None, outputs: vec![out] },
        "sin" => Operation::Sin { input: ids[0], options: None, outputs: vec![out] },
        "cos" => Operation::Cos { input: ids[0], options: None, outputs: vec![out] },
        "ceil" => Operation::Ceil { input: ids[0], options: None, outputs: vec![out] },
        "floor" => Operation::Floor { input: ids[0], options: None, outputs: vec![out] },
        "tan" => Operation::Tan { input: ids[0], options: None, outputs: vec![out] },
        "erf" => Operation::Erf { input: ids[0], options: None, outputs: vec![out] },
        "reciprocal" => Operation::Reciprocal { input: ids[0], options: None, outputs: vec![out] },
        "identity" => Operation::Identity { input: ids[0], options: None, outputs: vec![out] },
        "softplus" => Operation::Softplus { input: ids[0], options: None, outputs: vec![out] },
        "softsign" => Operation::Softsign { input: ids[0], options: None, outputs: vec![out] },
        "gelu" => Operation::Gelu { input: ids[0], options: None, outputs: vec![out] },
        "hardSigmoid" => Operation::HardSigmoid { input: ids[0], options: None, outputs: vec![out] },
        "hardSwish" => Operation::HardSwish { input: ids[0], options: None, outputs: vec![out] },
        "max" => Operation::Max { a: ids[0], b: ids[1], options: None, outputs: vec![out] },
        "min" => Operation::Min { a: ids[0], b: ids[1], options: None, outputs: vec![out] },
        "matmul" => Operation::Matmul { a: ids[0], b: ids[1], options: None, outputs: vec![out] },
        "equal" => Operation::Equal { a: ids[0], b: ids[1], options: None, outputs: vec![out] },
        "greater" => Operation::Greater { a: ids[0], b: ids[1], options: None, outputs: vec![out] },
        "greaterOrEqual" => Operation::GreaterOrEqual { a: ids[0], b: ids[1], options: None, outputs: vec![out] },
        "lesser" => Operation::Lesser { a: ids[0], b: ids[1], options: None, outputs: vec![out] },
        "lesserOrEqual" => Operation::LesserOrEqual { a: ids[0], b: ids[1], options: None, outputs: vec![out] },
        "notEqual" => Operation::NotEqual { a: ids[0], b: ids[1], options: None, outputs: vec![out] },
        _ => return Err(format!("unsupported op '{}'", op)),
    })
}

pub fn compile_from_nodes(nodes: &[GraphNode], output_names: &[String]) -> Result<Vec<u8>, String> {
    let output_set: HashSet<&String> = output_names.iter().collect();
    let mut n2i: HashMap<String, u32> = HashMap::new();
    let mut ops: Vec<Operand> = Vec::new();
    let mut nxt = 0u32;
    let mut reg = |n: &str, dt: u32, sh: &[u32]| -> u32 {
        if let Some(&i) = n2i.get(n) { return i; }
        let id = nxt; nxt += 1; n2i.insert(n.to_string(), id);
        ops.push(Operand { kind: OperandKind::Output, name: Some(n.to_string()),
            descriptor: OperandDescriptor { data_type: to_dtype(dt),
                shape: sh.iter().map(|&d| Dimension::Static(d)).collect(), pending_permutation: vec![] }});
        id
    };
    for n in nodes { reg(&n.output, n.data_type, &n.shape); }
    let producers: HashSet<&str> = nodes.iter().map(|n| n.output.as_str()).collect();
    let mut in_set: HashSet<String> = HashSet::new();
    for n in nodes { for inp in &n.inputs { if !producers.contains(inp.as_str()) && !in_set.contains(inp) {
        in_set.insert(inp.clone()); reg(inp, n.data_type, &n.shape); }}}
    let inp_ids: Vec<u32> = in_set.iter().map(|n| n2i[n.as_str()]).collect();

    let mut operations = Vec::new();
    let mut out_ids = Vec::new();
    let mut consts: HashMap<u32, ConstantData> = HashMap::new();
    for n in nodes {
        let ids: Vec<u32> = n.inputs.iter().map(|inp| n2i.get(inp.as_str()).copied()
            .ok_or_else(|| format!("input '{}' not found", inp))).collect::<Result<_,_>>()?;
        let oid = n2i[&n.output];
        if output_set.contains(&n.output) { out_ids.push(oid); }
        if let Some(ref d) = n.data { consts.insert(oid, ConstantData { data: d.clone(), label: None }); }
        operations.push(make_op(&n.op, &ids, oid)?);
    }

    let in_idx: HashSet<u32> = inp_ids.iter().copied().collect();
    for o in ops.iter_mut() { if let Some(ref n) = o.name { if let Some(&i) = n2i.get(n) {
        if in_idx.contains(&i) && !output_set.contains(n) { o.kind = OperandKind::Input; }
        if consts.contains_key(&i) { o.kind = OperandKind::Constant; }}}}

    let gi = GraphInfo { operands: ops, input_operands: inp_ids, output_operands: out_ids,
        operations, constant_operand_ids_to_handles: consts,
        id_to_constant_tensor_operand_map: HashMap::new(), quantized: false };

    OnnxConverter.convert(&gi).map(|c| c.data).map_err(|e| format!("ONNX: {e}"))
}

static CACHE: OnceLock<std::sync::Mutex<HashMap<usize, Vec<u8>>>> = OnceLock::new();
fn cache() -> &'static std::sync::Mutex<HashMap<usize, Vec<u8>>> {
    CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

pub fn compile_cached(nodes: &[GraphNode], output_names: &[String], key: usize) -> Result<Vec<u8>, String> {
    let mut c = cache().lock().map_err(|e| format!("cache: {e}"))?;
    if let Some(b) = c.get(&key) { return Ok(b.clone()); }
    let b = compile_from_nodes(nodes, output_names)?;
    c.insert(key, b.clone()); Ok(b)
}

pub fn run(model_bytes: &[u8], inputs: &[(&str, &[u8])]) -> Result<RunResult, String> {
    let mut session = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Session::builder().map_err(|e| format!("ort builder: {e}"))?
            .commit_from_memory(model_bytes).map_err(|e| format!("ort load: {e}"))
    })).map_err(|_| "ORT failed".to_string())??;

    let mut vals: Vec<ort::session::SessionInputValue> = Vec::new();
    for info in session.inputs().iter() {
        let name = info.name();
        let (_, data) = inputs.iter().find(|(n, _)| *n == name)
            .ok_or_else(|| format!("missing input '{}'", name))?;
        let floats: Vec<f32> = data.chunks(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        let t = Value::from_array((vec![floats.len() as i64].as_slice(), floats))
            .map_err(|e| format!("input '{}': {e}", name))?;
        vals.push(t.into_dyn().into());
    }
    let outputs = session.run(vals.as_slice()).map_err(|e| format!("ort run: {e}"))?;
    let mut bufs = Vec::new();
    for (_, value) in outputs.iter() {
        let (_, data) = value.try_extract_tensor::<f32>().map_err(|e| format!("extract: {e}"))?;
        bufs.push(data.iter().flat_map(|f| f.to_le_bytes()).collect());
    }
    Ok(RunResult { outputs: bufs })
}

pub use rustnn;
pub use rustnn::mlcontext;
pub use rustnn::mlgraphbuilder;
pub use rustnn::operator_enums;
