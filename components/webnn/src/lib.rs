/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::AtomicUsize;
use std::sync::LazyLock;

use rustnn::{ConstantData, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation, DataType as RnnDtype};
use rustnn::graph::Dimension;
use rustnn::mlcontext::{
    MLContext as RnnCtx, MLContextOptions, MLPowerPreference,
    MLTensorDescriptor as RnnTensorDesc, MLGraph as RnnGraph,
};
use rustnn::mlgraphbuilder::MLGraphBuilder as RnnBuilder;
use rustnn::operator_enums::MLOperandDataType as RnnDtypeEnum;

// ── Public types ──

pub struct GraphNode {
    pub op: String, pub inputs: Vec<String>, pub output: String,
    pub data_type: u32, pub shape: Vec<u32>,
    pub attrs: HashMap<String, f64>, pub data: Option<Vec<u8>>,
}

pub struct RunResult { pub outputs: Vec<Vec<u8>> }

// ── Backend trait ──

pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn compile(&self, nodes: &[GraphNode], output_names: &[String]) -> Result<usize, String>;
    fn run(&self, graph_id: usize, inputs: &[(&str, &[u8])], output_names: &[String]) -> Result<RunResult, String>;
}

static BACKEND: OnceLock<Mutex<Box<dyn Backend>>> = OnceLock::new();

pub fn set_backend(b: Box<dyn Backend>) {
    BACKEND.get_or_init(|| Mutex::new(b));
}

fn with_backend<T>(f: impl FnOnce(&dyn Backend) -> T) -> T {
    f(BACKEND.get_or_init(|| Mutex::new(Box::new(RustnnBackend))).lock().unwrap().as_ref())
}

pub fn compile(nodes: &[GraphNode], output_names: &[String]) -> Result<usize, String> {
    with_backend(|b| b.compile(nodes, output_names))
}
pub fn run(graph_id: usize, inputs: &[(&str, &[u8])], output_names: &[String]) -> Result<RunResult, String> {
    with_backend(|b| b.run(graph_id, inputs, output_names))
}

// ── Async dispatch — non-blocking JS thread ──

use std::sync::{Arc, Condvar};
use std::thread;

pub struct Ticket {
    result: Arc<(Mutex<Option<Result<RunResult, String>>>, Condvar)>,
}

impl Ticket {
    pub fn flush(self) -> Result<RunResult, String> {
        let (lock, cvar) = &*self.result;
        let mut r = lock.lock().unwrap();
        if r.is_none() { r = cvar.wait(r).unwrap(); }
        r.take().unwrap()
    }
}

use std::sync::mpsc::{channel, Sender};

struct Task {
    graph_id: usize,
    inputs: Vec<(String, Vec<u8>)>,
    output_names: Vec<String>,
    ticket: Arc<(Mutex<Option<Result<RunResult, String>>>, Condvar)>,
}

static TASK_TX: OnceLock<Mutex<Sender<Task>>> = OnceLock::new();

fn ensure_worker() {
    TASK_TX.get_or_init(|| {
        let (tx, rx) = channel::<Task>();
        thread::Builder::new().name("webnn-worker".into()).spawn(move || {
            for task in rx {
                let inputs: Vec<(&str, &[u8])> = task.inputs.iter()
                    .map(|(n, d)| (n.as_str(), d.as_slice())).collect();
                let r = with_backend(|b| b.run(task.graph_id, &inputs, &task.output_names));
                let (lock, cvar) = &*task.ticket;
                *lock.lock().unwrap() = Some(r);
                cvar.notify_one();
            }
        }).unwrap();
        Mutex::new(tx)
    });
}

pub fn dispatch_async(graph_id: usize, inputs: Vec<(String, Vec<u8>)>, output_names: Vec<String>) -> Ticket {
    ensure_worker();
    let ticket = Ticket { result: Arc::new((Mutex::new(None), Condvar::new())) };
    TASK_TX.get().unwrap().lock().unwrap().send(Task { graph_id, inputs, output_names, ticket: ticket.result.clone() }).unwrap();
    ticket
}

pub fn flush(ticket: Ticket) -> Result<RunResult, String> { ticket.flush() }

// ── Shared converter: GraphNode[] → GraphInfo ──

fn to_dtype(v: u32) -> RnnDtype {
    match v { 0=>RnnDtype::Float32,1=>RnnDtype::Float16,2=>RnnDtype::Int32,3=>RnnDtype::Uint32,
        4=>RnnDtype::Int64,5=>RnnDtype::Uint64,6=>RnnDtype::Int8,7=>RnnDtype::Uint8,_=>RnnDtype::Float32 }
}

fn make_op(op: &str, ids: &[u32], out: u32) -> Result<Operation, String> {
    Ok(match op {
        "add"=>Operation::Add{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "sub"=>Operation::Sub{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "mul"=>Operation::Mul{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "div"=>Operation::Div{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "pow"=>Operation::Pow{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "relu"=>Operation::Relu{input:ids[0],options:None,outputs:vec![out]},
        "sigmoid"=>Operation::Sigmoid{input:ids[0],options:None,outputs:vec![out]},
        "tanh"=>Operation::Tanh{input:ids[0],options:None,outputs:vec![out]},
        "abs"=>Operation::Abs{input:ids[0],options:None,outputs:vec![out]},
        "neg"=>Operation::Neg{input:ids[0],options:None,outputs:vec![out]},
        "exp"=>Operation::Exp{input:ids[0],options:None,outputs:vec![out]},
        "log"=>Operation::Log{input:ids[0],options:None,outputs:vec![out]},
        "sqrt"=>Operation::Sqrt{input:ids[0],options:None,outputs:vec![out]},
        "sin"=>Operation::Sin{input:ids[0],options:None,outputs:vec![out]},
        "cos"=>Operation::Cos{input:ids[0],options:None,outputs:vec![out]},
        "ceil"=>Operation::Ceil{input:ids[0],options:None,outputs:vec![out]},
        "floor"=>Operation::Floor{input:ids[0],options:None,outputs:vec![out]},
        "tan"=>Operation::Tan{input:ids[0],options:None,outputs:vec![out]},
        "erf"=>Operation::Erf{input:ids[0],options:None,outputs:vec![out]},
        "reciprocal"=>Operation::Reciprocal{input:ids[0],options:None,outputs:vec![out]},
        "identity"=>Operation::Identity{input:ids[0],options:None,outputs:vec![out]},
        "softplus"=>Operation::Softplus{input:ids[0],options:None,outputs:vec![out]},
        "softsign"=>Operation::Softsign{input:ids[0],options:None,outputs:vec![out]},
        "gelu"=>Operation::Gelu{input:ids[0],options:None,outputs:vec![out]},
        "hardSigmoid"=>Operation::HardSigmoid{input:ids[0],options:None,outputs:vec![out]},
        "hardSwish"=>Operation::HardSwish{input:ids[0],options:None,outputs:vec![out]},
        "max"=>Operation::Max{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "min"=>Operation::Min{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "matmul"=>Operation::Matmul{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "equal"=>Operation::Equal{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "greater"=>Operation::Greater{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "greaterOrEqual"=>Operation::GreaterOrEqual{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "lesser"=>Operation::Lesser{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "lesserOrEqual"=>Operation::LesserOrEqual{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "notEqual"=>Operation::NotEqual{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "transpose"=>Operation::Transpose{input:ids[0],options:None,outputs:vec![out]},
        "logicalNot"=>Operation::LogicalNot{input:ids[0],options:None,outputs:vec![out]},
        "logicalAnd"=>Operation::LogicalAnd{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "logicalOr"=>Operation::LogicalOr{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "logicalXor"=>Operation::LogicalXor{a:ids[0],b:ids[1],options:None,outputs:vec![out]},
        "reshape"=>Operation::Reshape{input:ids[0],new_shape:vec![],options:None,outputs:vec![out]},
        "concat"=>Operation::Concat{inputs:ids.to_vec(),axis:0,options:None,outputs:vec![out]},
        "softmax"=>Operation::Softmax{input:ids[0],axis:1,options:None,outputs:vec![out]},
        "clamp"=>Operation::Clamp{input:ids[0],options:None,outputs:vec![out]},
        "prelu"=>Operation::Prelu{input:ids[0],slope:ids[1],options:None,outputs:vec![out]},
        "elu"=>Operation::Elu{input:ids[0],options:None,outputs:vec![out]},
        "leakyRelu"=>Operation::LeakyRelu{input:ids[0],options:None,outputs:vec![out]},
        _=>return Err(format!("unsupported op '{}'",op)),
    })
}

pub fn nodes_to_graph_info(nodes: &[GraphNode], output_names: &[String]) -> Result<GraphInfo, String> {
    let output_set: HashSet<&String> = output_names.iter().collect();
    let mut n2i: HashMap<String, u32> = HashMap::new();
    let mut ops: Vec<Operand> = Vec::new();
    let mut nxt = 0u32;
    let mut reg = |n:&str,dt:u32,sh:&[u32]|->u32{
        if let Some(&i)=n2i.get(n){return i}
        let id=nxt;nxt+=1;n2i.insert(n.to_string(),id);
        ops.push(Operand{kind:OperandKind::Output,name:Some(n.to_string()),
            descriptor:OperandDescriptor{data_type:to_dtype(dt),
                shape:sh.iter().map(|&d|Dimension::Static(d)).collect(),pending_permutation:vec![]}});
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
        let oid = n2i[&n.output];
        if output_set.contains(&n.output) { out_ids.push(oid); }
        if let Some(ref d) = n.data { consts.insert(oid, ConstantData { data: d.clone(), label: None }); }
        if n.op == "constant" { continue; }
        let ids: Vec<u32> = n.inputs.iter().map(|inp| n2i.get(inp.as_str()).copied()
            .ok_or_else(|| format!("input '{}' not found", inp))).collect::<Result<_,_>>()?;
        operations.push(make_op(&n.op, &ids, oid)?);
    }

    let in_idx: HashSet<u32> = inp_ids.iter().copied().collect();
    for o in ops.iter_mut() { if let Some(ref n) = o.name { if let Some(&i) = n2i.get(n) {
        if in_idx.contains(&i) && !output_set.contains(n) { o.kind = OperandKind::Input; }
        if consts.contains_key(&i) { o.kind = OperandKind::Constant; }}}}
    for o in ops.iter_mut() { if o.descriptor.data_type == RnnDtype::Uint8 { o.descriptor.data_type = RnnDtype::Float32; } }

    Ok(GraphInfo { operands: ops, input_operands: inp_ids, output_operands: out_ids,
        operations, constant_operand_ids_to_handles: consts,
        id_to_constant_tensor_operand_map: HashMap::new(), quantized: false })
}

// ── Rustnn Backend ──

struct RustnnBackend;

static G_GRAPHS: OnceLock<std::sync::Mutex<HashMap<usize, RnnGraph<'static>>>> = OnceLock::new();
static NEXT_GID: AtomicUsize = AtomicUsize::new(1);

impl Backend for RustnnBackend {
    fn name(&self) -> &str { "rustnn" }

    fn compile(&self, nodes: &[GraphNode], output_names: &[String]) -> Result<usize, String> {
        let gi = nodes_to_graph_info(nodes, output_names)?;
        let opts = MLContextOptions::new(MLPowerPreference::Default, true);
        let mut ctx = RnnCtx::create(&opts).map_err(|e| format!("context: {e:?}"))?;
        let mut builder = RnnBuilder::new(&mut ctx).map_err(|e| format!("builder: {e:?}"))?;
        let graph = builder.build_graph_info(gi).map_err(|e| format!("build: {e:?}"))?;
        let gid = NEXT_GID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        G_GRAPHS.get_or_init(|| std::sync::Mutex::new(HashMap::new())).lock().unwrap().insert(gid, graph);
        Ok(gid)
    }

    fn run(&self, graph_id: usize, inputs: &[(&str, &[u8])], output_names: &[String]) -> Result<RunResult, String> {
        let mut graphs = G_GRAPHS.get_or_init(|| std::sync::Mutex::new(HashMap::new())).lock().unwrap();
        let mut graph = graphs.remove(&graph_id).ok_or("graph not found")?;
        drop(graphs);

        let opts = MLContextOptions::new(MLPowerPreference::Default, true);
        let mut ctx = RnnCtx::create(&opts).map_err(|e| format!("context: {e:?}"))?;

        let mut in_tensors: Vec<(String, rustnn::mlcontext::MLTensor)> = Vec::new();
        let mut out_tensors: Vec<(String, rustnn::mlcontext::MLTensor)> = Vec::new();

        for &(name, data) in inputs {
            let floats: Vec<f32> = data.chunks(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
            let mut desc = RnnTensorDesc::new(RnnDtypeEnum::Float32, vec![floats.len() as u64]);
            desc.set_readable(true); desc.set_writable(true);
            let rt = ctx.create_tensor(&desc).map_err(|e| format!("create: {e:?}"))?;
            ctx.write_tensor(&rt, &floats).map_err(|e| format!("write: {e:?}"))?;
            in_tensors.push((name.to_string(), rt));
            let mut od = RnnTensorDesc::new(RnnDtypeEnum::Float32, vec![floats.len() as u64]);
            od.set_readable(true); od.set_writable(true);
            let ot = ctx.create_tensor(&od).map_err(|e| format!("create out: {e:?}"))?;
            out_tensors.push((name.to_string(), ot));
        }

        let ri: HashMap<&str, &rustnn::mlcontext::MLTensor> = in_tensors.iter()
            .map(|(n, t)| (n.as_str(), t)).collect();
        let ro: HashMap<&str, &rustnn::mlcontext::MLTensor> = out_tensors.iter()
            .zip(output_names.iter())
            .map(|((_, t), n)| (n.as_str(), t)).collect();

        ctx.dispatch(&mut graph, &ri, &ro).map_err(|e| format!("dispatch: {e:?}"))?;

        let mut outputs = Vec::new();
        for (_, ot) in &out_tensors {
            let sz: usize = ot.shape().iter().product::<u64>() as usize;
            let mut out = vec![0.0f32; sz];
            ctx.read_tensor(ot, &mut out).map_err(|e| format!("read: {e:?}"))?;
            outputs.push(out.iter().flat_map(|f| f.to_le_bytes()).collect());
        }

        G_GRAPHS.get_or_init(|| std::sync::Mutex::new(HashMap::new())).lock().unwrap().insert(graph_id, graph);
        Ok(RunResult { outputs })
    }
}

// ── Mock Backend ──

pub struct MockBackend;

impl Backend for MockBackend {
    fn name(&self) -> &str { "mock" }
    fn compile(&self, _nodes: &[GraphNode], _output_names: &[String]) -> Result<usize, String> {
        Ok(0) // always returns graph_id 0
    }
    fn run(&self, _graph_id: usize, inputs: &[(&str, &[u8])], _output_names: &[String]) -> Result<RunResult, String> {
        // Copy first input to all outputs
        let data = inputs.first().map(|(_, d)| d.to_vec()).unwrap_or_default();
        let count = _output_names.len().max(1);
        Ok(RunResult { outputs: vec![data; count] })
    }
}

pub use rustnn;
pub use rustnn::mlcontext;
pub use rustnn::mlgraphbuilder;
pub use rustnn::operator_enums;
