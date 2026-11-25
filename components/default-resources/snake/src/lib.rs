/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! A tiny Snake game used by the network error page.
//!
//! This is deliberately dependency-free and `no_std`: it compiles to a few
//! kilobytes of wasm and owns the game state itself. The page renders the
//! grid on a 2D canvas and feeds input back through `snake_set_dir`.
//!
//! The wasm is checked in as `../resources/snake.wasm` and is only included in
//! Servo builds with the `snake-game` feature. To rebuild it after changing
//! this file:
//!
//! ```text
//! cargo build --release --target wasm32-unknown-unknown
//! install -m 644 target/wasm32-unknown-unknown/release/snake_game.wasm ../resources/snake.wasm
//! ```

#![no_std]

use core::ptr::addr_of_mut;

const W: usize = 20;
const H: usize = 20;
const CELLS: usize = W * H;

const EMPTY: u8 = 0;
const BODY: u8 = 1;
const HEAD: u8 = 2;
const FOOD: u8 = 3;

struct Game {
    grid: [u8; CELLS],
    body: [u16; CELLS],
    head: usize,
    len: usize,
    dir: u8,
    next_dir: u8,
    score: u32,
    rng: u32,
    alive: bool,
    full: bool,
}

impl Game {
    const fn new() -> Self {
        Self {
            grid: [EMPTY; CELLS],
            body: [0; CELLS],
            head: 0,
            len: 0,
            dir: 1,
            next_dir: 1,
            score: 0,
            rng: 1,
            alive: false,
            full: false,
        }
    }

    fn reset(&mut self, seed: u32, dir: u8) {
        let dir = dir % 4;
        self.grid = [EMPTY; CELLS];
        self.body = [0; CELLS];
        self.len = 5;
        self.head = 4;
        self.dir = dir;
        self.next_dir = dir;
        self.score = 0;
        self.alive = true;
        self.full = false;
        self.rng = if seed == 0 { 0x9e37_79b9 } else { seed };

        let cx = W / 2;
        let cy = H / 2;
        let (dx, dy) = match dir {
            0 => (0i32, -1i32),
            1 => (1, 0),
            2 => (0, 1),
            _ => (-1, 0),
        };
        // body[0] is the tail, body[4] is the head. This matches the five
        // segments of the Servo logo that uncoil into the snake.
        for i in 0..5 {
            let x = (cx as i32 - dx * (4 - i as i32)) as usize;
            let y = (cy as i32 - dy * (4 - i as i32)) as usize;
            let cell = y * W + x;
            self.body[i] = cell as u16;
            self.grid[cell] = if i == 4 { HEAD } else { BODY };
        }
        self.place_food();
    }

    fn place_food(&mut self) {
        let mut attempts = 0usize;
        loop {
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 17;
            self.rng ^= self.rng << 5;
            let idx = (self.rng as usize) % CELLS;
            if self.grid[idx] == EMPTY {
                self.grid[idx] = FOOD;
                return;
            }
            attempts += 1;
            if attempts > CELLS * 2 {
                self.full = true;
                return;
            }
        }
    }

    /// Reorient the snake for its first move without moving the head or
    /// re-placing the food. Returns 0 when the direction is invalid or the new
    /// body would overlap the food.
    fn start(&mut self, dir: u8) -> u32 {
        if dir > 3 || !self.alive {
            return 0;
        }
        let head_cell = self.body[self.head] as usize;
        let hx = (head_cell % W) as isize;
        let hy = (head_cell / W) as isize;
        let (dx, dy) = match dir {
            0 => (0isize, -1isize),
            1 => (1, 0),
            2 => (0, 1),
            _ => (-1, 0),
        };

        // cells[0] is the head, followed by the body stretching behind it.
        let mut cells = [0u16; CELLS];
        for i in 0..self.len {
            let x = hx - dx * (i as isize);
            let y = hy - dy * (i as isize);
            if x < 0 || x >= W as isize || y < 0 || y >= H as isize {
                return 0;
            }
            let cell = (y as usize) * W + (x as usize);
            if self.grid[cell] == FOOD {
                return 0;
            }
            cells[i] = cell as u16;
        }

        for i in 0..self.len {
            self.grid[self.body[i] as usize] = EMPTY;
        }
        for i in 0..self.len {
            let cell = cells[self.len - 1 - i];
            self.body[i] = cell;
            self.grid[cell as usize] = if i == self.len - 1 { HEAD } else { BODY };
        }
        self.head = self.len - 1;
        self.dir = dir;
        self.next_dir = dir;
        1
    }

    fn set_dir(&mut self, dir: u8) {
        if dir > 3 {
            return;
        }
        // Compare against the direction of the last move, not the queued
        // direction: otherwise "up, then left" between two ticks could turn a
        // right-moving snake straight around.
        if (dir + 2) % 4 == self.dir {
            return;
        }
        self.next_dir = dir;
    }

    /// Returns 0 for a normal move, 1 for game over, 2 for eating food (or a win).
    fn step(&mut self) -> u32 {
        if !self.alive {
            return 1;
        }
        self.dir = self.next_dir;
        let head_cell = self.body[self.head] as usize;
        let hx = (head_cell % W) as isize;
        let hy = (head_cell / W) as isize;
        let (nx, ny) = match self.dir {
            0 => (hx, hy - 1),
            1 => (hx + 1, hy),
            2 => (hx, hy + 1),
            _ => (hx - 1, hy),
        };
        if nx < 0 || nx >= W as isize || ny < 0 || ny >= H as isize {
            self.alive = false;
            return 1;
        }

        let new_cell = (ny as usize) * W + (nx as usize);
        let growing = self.grid[new_cell] == FOOD;
        let tail_idx = (self.head + CELLS - (self.len - 1)) % CELLS;
        let tail_cell = self.body[tail_idx] as usize;
        let occupied = self.grid[new_cell] == BODY || self.grid[new_cell] == HEAD;
        // The tail cell is vacating in the same tick unless the snake is growing.
        if occupied && !(!growing && new_cell == tail_cell) {
            self.alive = false;
            return 1;
        }
        if self.len >= CELLS {
            self.alive = false;
            self.full = true;
            return 2;
        }

        let old_head_cell = head_cell;
        let new_head_idx = (self.head + 1) % CELLS;
        if !growing {
            self.grid[tail_cell] = EMPTY;
        }
        self.body[new_head_idx] = new_cell as u16;
        self.head = new_head_idx;
        self.grid[old_head_cell] = BODY;
        self.grid[new_cell] = HEAD;

        if growing {
            self.len += 1;
            self.score += 1;
            self.place_food();
            if self.full {
                self.alive = false;
                return 2;
            }
            return 2;
        }
        0
    }
}

static mut GAME: Game = Game::new();

fn game() -> &'static mut Game {
    // The game is only touched from the single page main thread.
    unsafe { &mut *addr_of_mut!(GAME) }
}

/// Resets the game. `dir` is 0 = up, 1 = right, 2 = down, 3 = left.
#[no_mangle]
pub extern "C" fn snake_reset(seed: u32, dir: u32) {
    game().reset(seed, dir as u8);
}

/// Orient the snake for its first move, keeping the food in place.
#[no_mangle]
pub extern "C" fn snake_start(dir: u32) -> u32 {
    game().start(dir as u8)
}

#[no_mangle]
pub extern "C" fn snake_set_dir(dir: u32) {
    game().set_dir(dir as u8);
}

#[no_mangle]
pub extern "C" fn snake_step() -> u32 {
    game().step()
}

/// Pointer to a `W * H` byte grid: 0 = empty, 1 = body, 2 = head, 3 = food.
#[no_mangle]
pub extern "C" fn snake_grid() -> *const u8 {
    game().grid.as_ptr()
}

#[no_mangle]
pub extern "C" fn snake_score() -> u32 {
    game().score
}

#[no_mangle]
pub extern "C" fn snake_alive() -> u32 {
    game().alive as u32
}

#[no_mangle]
pub extern "C" fn snake_len() -> u32 {
    game().len as u32
}

/// The direction the next tick will use (useful for tests and debugging).
#[no_mangle]
pub extern "C" fn snake_next_dir() -> u32 {
    game().next_dir as u32
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}
