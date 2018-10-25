package fpinscala.state
/*
Inserting a coin into a locked machine will cause it to unlock if there is any candy left.
Turning the knob on an unlocked machine will cause it to dispense candy and become locked.
Turning the knob on a locked machine or inserting a coin into an unlocked machine does nothing.
A machine that is out of candy ignores all inputs.
*/

sealed trait Input
case object Coin extends Input
case object Turn extends Input
case class Machine(locked: Boolean, candies: Int, coins: Int) {
  def simulateMachine(inputs: List[Input]): State[Machine, Int] =
    State(_ => inputs.foldRight((this.coins, this))((inp, s) => s._2.operateInput(inp)))
  def run(inputs: List[Input]): (Int, Machine) =
    inputs.foldLeft((this.coins, this))((s, inp) => s._2.operateInput(inp))

  def operateInput(input: Input): (Int, Machine) =
    this match {
      case Machine(_, 0, _) => (0, this)
      case Machine(false, _, _) =>
        input match {
          case Coin => (this.coins, this) // insert coin into unlocked machine
          case Turn => (this.coins, Machine(locked = true, candies = this.candies - 1, coins = this.coins)) // give candy
        }
      case Machine(true, _, _) =>
        input match {
          case Coin => (this.coins + 1, Machine(locked = false, candies = this.candies, coins = this.coins + 1)) // unlock machine
          case Turn => (this.coins, this) // turn the knob of locked machine
        }
    }
}