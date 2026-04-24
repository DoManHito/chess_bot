use pyo3::prelude::*;
use rayon::prelude::*;
use chess::{Board, MoveGen};
use std::collections::HashMap;
use std::str::FromStr;

#[pyclass]
pub struct MCTSNode {
    #[pyo3(get)]
    pub chess_move: Option<String>,
    
    #[pyo3(get)]
    pub children: HashMap<String, Py<MCTSNode>>,
    
    #[pyo3(get, set)]
    pub n: u32,
    #[pyo3(get, set)]
    pub w: f32,
    #[pyo3(get, set)]
    pub q: f32,
    #[pyo3(get, set)]
    pub p: f32, 
    #[pyo3(get, set)]
    pub is_terminal: bool,
    #[pyo3(get, set)]
    pub term_value: Option<f32>,
}

#[pymethods]
impl MCTSNode {
    #[new]
    #[pyo3(signature = (prior, chess_move=None))]
    pub fn new(prior: f32, chess_move: Option<String>) -> Self {
        Self {
            chess_move,
            children: HashMap::new(),
            n: 0,
            w: 0.0,
            q: 0.0,
            p: prior,
            is_terminal: false,
            term_value: None,
        }
    }

    pub fn children_moves(&self) -> Vec<String> {
        self.children.keys().cloned().collect()
    }

    pub fn select_child_move(&self, py: Python<'_>, c_puct: f32) -> Option<String> {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_move = None;
        let sqrt_n = (self.n as f32 + 1e-8).sqrt();

        for (m_str, child_py) in &self.children {
            let child = child_py.bind(py).borrow();
            let u_score = if child.is_terminal && child.term_value.map_or(false, |v| v < -0.99) {
                10000.0
            } else {
                -child.q + c_puct * child.p * (sqrt_n / (1.0 + child.n as f32))
            };
            if u_score > best_score {
                best_score = u_score;
                best_move = Some(m_str.clone());
            }
        }
        best_move
    }

    pub fn get_child(&self, py: Python<'_>, m_str: String) -> Option<Py<MCTSNode>> {
        self.children.get(&m_str).map(|node_py| node_py.clone_ref(py))
    }

    pub fn expand(&mut self, py: Python<'_>, fen: String, policy: HashMap<String, f32>) {
        let board = Board::from_str(&fen).unwrap_or_else(|_| Board::default());
        let move_gen = MoveGen::new_legal(&board);
        for m in move_gen {
            let m_str = m.to_string();
            let prior = *policy.get(&m_str).unwrap_or(&0.0);
            let child_node = Py::new(py, MCTSNode::new(prior, Some(m_str.clone()))).unwrap();
            self.children.insert(m_str, child_node);
        }
    }

    pub fn update(&mut self, value: f32) {
        self.n += 1;
        self.w += value;
        self.q = self.w / (self.n as f32);
    }
}

#[pyfunction]
pub fn batch_select_moves(nodes: Vec<Py<MCTSNode>>, c_puct: f32) -> Vec<Option<String>> {
    struct NodeData {
        n: u32,
        children: Vec<(String, u32, f32, f32, bool, Option<f32>)>,
    }
    let data: Vec<NodeData> = Python::with_gil(|py| {
        nodes.iter().map(|node_py| {
            let n = node_py.bind(py).borrow();
            let mut children_data = Vec::new();
            for (m_str, child_py) in &n.children {
                let c = child_py.bind(py).borrow();
                children_data.push((m_str.clone(), c.n, c.q, c.p, c.is_terminal, c.term_value));
            }
            NodeData { n: n.n, children: children_data }
        }).collect()
    });
    data.into_par_iter().map(|node| {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_move = None;
        let sqrt_n = (node.n as f32 + 1e-8).sqrt();
        for (m_str, c_n, c_q, c_p, c_is_term, c_term_val) in node.children {
            let u_score = if c_is_term && c_term_val == Some(-1.0) { 10000.0 } 
                          else { -c_q + c_puct * c_p * (sqrt_n / (1.0 + c_n as f32)) };
            if u_score > best_score {
                best_score = u_score;
                best_move = Some(m_str);
            }
        }
        best_move
    }).collect()
}

#[pymodule]
fn mcts_engine(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> { 
    m.add_class::<MCTSNode>()?;
    m.add_function(wrap_pyfunction!(batch_select_moves, m)?)?;
    Ok(())
}