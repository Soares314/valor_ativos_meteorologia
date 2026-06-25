using Godot;
using System.Linq;

public partial class TreinarButton : Button
{
	private bool[] Preenchido = new bool[4];

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
		if(Preenchido.All(x => x))
		{
			Disabled = false;
		}
	}

	private void OnLocalTextChanged()
	{
		Preenchido[0] = true;
	}

	private void OnDataTextChanged()
	{
		Preenchido[1] = true;
	}
}
